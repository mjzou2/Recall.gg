"""LLM analysis service for session summaries and chunk tagging.

Uses Claude Haiku to analyze transcribed scrim comms. Generates a session
summary and per-chunk tags for notable moments (objective calls, shotcalls,
disagreements, good comms, silences, tilt, info sharing).

Fault-tolerant: returns None on any failure, logs warning. Never blocks
the processing pipeline.
"""

import json
import logging
from typing import Dict, List, Optional

from app.config import ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

_client = None
_client_init_attempted = False

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 16384

VALID_TAG_TYPES = {
    "objective_call", "shotcall", "disagreement", "good_comm",
    "silence", "tilt", "info_share",
}

SYSTEM_PROMPT_TEMPLATE = """\
You are analyzing a League of Legends scrim voice comms transcript. Your audience is a coach reviewing this scrim. Every chunk has an ID, start/end timestamps in milliseconds, a speaker label, and transcribed text.

The total session duration is {duration_ms}ms.

REFERENCE - League of Legends terms that appear in scrim comms:
Objectives: baron, nashor, nash, elder, elder drake, atakhan, dragon, drake, dragon soul, soul, infernal, mountain, ocean, cloud, chemtech, hextech, herald, rift herald, grubs, voidgrubs, scuttle, red buff, blue buff, gromp, krugs, raptors, wolves
Locations: alcove, banana bush, baron pit, base, blast cone, blue buff, blue side, bot lane, brush, bush, dragon pit, fountain, gromp, honeyfruit, inhib tower, jungle, krugs, mid lane, nexus towers, pixel bush, raptors, red buff, red side, river, scryer's bloom, side lane, tier 1, tier 2, tier 3, top lane, tribush, wolves

TASK 1: SUMMARY (2-4 sentences)

Answer these three questions in paragraph form:
- Shotcalling: Which speaker(s) initiated the most objective calls and team directions? Was shotcalling concentrated in one voice or spread across multiple?
- Objectives: What objectives were discussed (baron, dragon, towers, etc)? How frequently were objectives part of the comms?
- Breakdowns: Were there conflicting calls from different speakers, moments of confusion about what to do, or extended silences during mid-to-late game?

TASK 2: TAGS

Tag chunks that are notable. Do NOT tag every chunk. Only tag chunks where something meaningful happened. Return an array of objects with chunk_id, type, and label.

Tag types and what to look for:

- "objective_call": A speaker tells the team to take or set up for baron, dragon, elder, herald, tower, or inhibitor.
- "shotcall": A speaker gives a clear team direction that is NOT an objective. Examples: "group mid," "split top," "back off," "fight here," "dive bot."
- "disagreement": Two speakers give conflicting directions within a short time window. Example: Speaker 1 says "go baron" while Speaker 2 says "we can't, rotate bot." This is high value for coaches.
- "good_comm": Clean, clear coordination. Multiple speakers confirming the same plan, sharing cooldown info, or building on each other's calls.
- "silence": A gap of 30+ seconds between chunks during mid-to-late game (after 10:00). Indicates the team stopped communicating during a period where they probably should have been talking. Use the timestamps to detect this, not the text.
- "tilt": Frustration, blame, negativity, or giving up. Examples: "why did you do that," "this is over," "I pinged you three times."
- "info_share": A speaker shares tactical information. Examples: "flash is down on their mid," "they're doing raptors," "TP is up in 20 seconds."

For the label field, write a short human-readable description of what happened (10 words max).

RESPONSE FORMAT

Return ONLY valid JSON, no markdown, no backticks:

{{
  "summary": "string",
  "tags": [
    {{"chunk_id": "string", "type": "string", "label": "string"}}
  ]
}}

If no chunks deserve a particular tag type, simply don't include any tags of that type. An empty tags array is fine if the session has nothing notable."""


def _get_client():
    """Lazy-init the Anthropic client."""
    global _client, _client_init_attempted
    if _client_init_attempted:
        return _client
    _client_init_attempted = True
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set, LLM analysis disabled")
        return None
    try:
        import anthropic
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("Initialized Anthropic client for model: %s", MODEL)
    except Exception as exc:
        logger.warning("Failed to initialize Anthropic client: %s", exc)
        _client = None
    return _client


def _format_chunks(chunks: List[Dict]) -> str:
    """Format chunk data as chunk_id | start_ms-end_ms | speaker | text."""
    lines = []
    for chunk in chunks:
        speaker = chunk.get("speaker") or "unknown"
        lines.append(
            f"{chunk['id']} | {chunk['start_ms']}-{chunk['end_ms']} | {speaker} | {chunk['text']}"
        )
    return "\n".join(lines)


def analyze_session(chunks: List[Dict], duration_ms: int) -> Optional[Dict]:
    """Analyze session chunks with Claude Haiku.

    Args:
        chunks: List of dicts with keys: id, start_ms, end_ms, text, speaker
        duration_ms: Total session duration in milliseconds

    Returns:
        {"summary": str, "tags": [{"chunk_id": str, "type": str, "label": str}]}
        or None on failure.
    """
    if not chunks:
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        transcript = _format_chunks(chunks)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(duration_ms=duration_ms)

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": transcript}],
        )

        if response.stop_reason == "max_tokens":
            logger.warning("LLM response truncated (hit max_tokens=%d)", MAX_TOKENS)
            return None

        raw_text = response.content[0].text.strip()

        # Strip markdown fences if present (defensive)
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3].strip()

        result = json.loads(raw_text)

        if not isinstance(result, dict):
            logger.warning("LLM response is not a dict: %s", type(result))
            return None
        if "summary" not in result or "tags" not in result:
            logger.warning("LLM response missing required keys: %s", list(result.keys()))
            return None
        if not isinstance(result["tags"], list):
            logger.warning("LLM response 'tags' is not a list")
            return None

        # Validate tags against actual chunk IDs and allowed types
        chunk_ids = {c["id"] for c in chunks}
        valid_tags = []
        for tag in result["tags"]:
            if (isinstance(tag, dict)
                    and tag.get("chunk_id") in chunk_ids
                    and tag.get("type") in VALID_TAG_TYPES
                    and isinstance(tag.get("label"), str)
                    and tag["label"].strip()):
                valid_tags.append({
                    "chunk_id": tag["chunk_id"],
                    "type": tag["type"],
                    "label": tag["label"].strip(),
                })

        summary = str(result["summary"]).strip()

        logger.info(
            "LLM analysis complete: summary=%d chars, tags=%d/%d valid",
            len(summary), len(valid_tags), len(result["tags"]),
        )

        return {"summary": summary, "tags": valid_tags}

    except json.JSONDecodeError as exc:
        logger.warning("LLM response is not valid JSON: %s", exc)
        return None
    except Exception as exc:
        logger.warning("LLM analysis failed: %s", exc)
        return None
