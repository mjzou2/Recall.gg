"""LLM analysis service for session scorecard and chunk tagging.

Uses Claude Haiku to analyze transcribed scrim comms. Generates a scorecard
(five comms categories rated 1-10) and per-chunk tags for notable moments
(objective calls, shotcalls, disagreements, silences, tilt).

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

VALID_SCORE_CATEGORIES = {
    "summoner_tracking", "objective_setup", "teamfight_comms",
    "shotcall_clarity", "map_awareness",
}

VALID_TAG_TYPES = {
    "objective_call", "shotcall", "disagreement",
    "silence", "tilt",
}

SYSTEM_PROMPT_TEMPLATE = """\
You are analyzing a League of Legends scrim voice comms transcript. Your audience is a coach reviewing this scrim. Every chunk has an ID, start/end timestamps in milliseconds, a speaker label, and transcribed text.

The total session duration is {duration_ms}ms.

REFERENCE - League of Legends terms that appear in scrim comms:
Objectives: baron, nashor, nash, elder, elder drake, atakhan, dragon, drake, dragon soul, soul, infernal, mountain, ocean, cloud, chemtech, hextech, herald, rift herald, grubs, voidgrubs, scuttle, red buff, blue buff, gromp, krugs, raptors, wolves
Locations: alcove, banana bush, baron pit, base, blast cone, blue buff, blue side, bot lane, brush, bush, dragon pit, fountain, gromp, honeyfruit, inhib tower, jungle, krugs, mid lane, nexus towers, pixel bush, raptors, red buff, red side, river, scryer's bloom, side lane, tier 1, tier 2, tier 3, top lane, tribush, wolves

TASK 1: SCORECARD

Rate this session in each category from 1-10. For each score, include the chunk_ids that most influenced your rating (only the clearest examples, not every instance).

SUMMONER TRACKING (calling enemy cooldowns - flash, TP, ults, ignite, etc):
1-3: Team rarely mentions enemy summoner spells. Key cooldowns go untracked.
4-6: Some summoner calls but inconsistent. Team tracks obvious ones (flash) but misses others.
7-9: Regular callouts on key cooldowns. Multiple speakers contributing tracking info.
10: Comprehensive tracking throughout. Team proactively shares timers and plays around enemy cooldowns.

OBJECTIVE SETUP (communication before baron, dragon, grubs, herald):
1-3: Objectives taken with little or no prior discussion. Team walks in without a plan.
4-6: Some objective calls but setup is disorganized. Calls happen but conditions aren't discussed.
7-9: Clear objective calls with discussion of conditions (enemy cooldowns, wave state, vision). Team aligns before committing.
10: Every objective is planned with full context. Team discusses win conditions, assigns roles, and has contingency if contested.

TEAMFIGHT COMMS (communication during fights):
1-3: Team goes quiet during fights or comms become chaotic and overlapping.
4-6: Some target calling or peel requests but not consistent. Calls happen late or get lost.
7-9: Clear target calling, cooldown sharing during fights, focus fire coordination.
10: Crisp real-time coordination. Target switches communicated, cooldowns tracked mid-fight, disengage calls are clean.

SHOTCALL CLARITY (are team directions clear and unified):
1-3: No clear shotcaller. Calls are vague, conflicting, or absent. Team moves without direction.
4-6: Shotcalling exists but sometimes conflicting or hesitant. Multiple voices giving different directions.
7-9: One or two clear voices driving decisions. Team generally follows calls with minimal confusion.
10: Decisive, unified shotcalling. Calls are specific, timely, and the team executes together.

MAP AWARENESS (sharing enemy positions, jungle tracking, rotation calls, lane status):
1-3: Team rarely communicates enemy positions or lane state. Rotations happen without warning.
4-6: Some callouts but reactive rather than proactive. Team shares info after getting caught, not before.
7-9: Regular proactive calls on enemy movement, jungle pathing, rotation timing, and lane state (examples: "I have prio," "I need help crashing," "I'm gankable," "wave is pushing to me").
10: Constant information flow. Team tracks enemy jungler, calls missing laners early, anticipates rotations, and keeps teammates updated on lane pressure and vulnerability.

TASK 2: TAGS

Tag ONLY moments a coach would want to review. Most chunks are routine comms and should NOT be tagged. When in doubt, do not tag. Return an array of objects with chunk_id, type, and label.

Tag types:

- "objective_call": A speaker tells the team to take or set up for baron, dragon, elder, herald, tower, or inhibitor.
- "shotcall": A speaker gives a clear team direction that is NOT an objective. Examples: "group mid," "split top," "back off," "fight here," "dive bot."
- "disagreement": Two speakers give conflicting directions within a short time window. Example: Speaker 1 says "go baron" while Speaker 2 says "we can't, rotate bot." This is high value for coaches.
- "silence": A gap of 30+ seconds between chunks during mid-to-late game (after 10:00). Indicates the team stopped communicating during a period where they probably should have been talking. Use the timestamps to detect this, not the text.
- "tilt": Frustration, blame, negativity, or giving up. Examples: "why did you do that," "this is over," "I pinged you three times."

For the label field, write a short human-readable description of what happened (10 words max).

RESPONSE FORMAT

Return ONLY valid JSON, no markdown, no backticks:

{{
  "scores": {{
    "summoner_tracking": {{"score": 0, "chunk_ids": []}},
    "objective_setup": {{"score": 0, "chunk_ids": []}},
    "teamfight_comms": {{"score": 0, "chunk_ids": []}},
    "shotcall_clarity": {{"score": 0, "chunk_ids": []}},
    "map_awareness": {{"score": 0, "chunk_ids": []}}
  }},
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
        {"scores": {category: {"score": int, "chunk_ids": [str]}},
         "tags": [{"chunk_id": str, "type": str, "label": str}]}
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
        if "scores" not in result or not isinstance(result["scores"], dict):
            logger.warning("LLM response missing 'scores' dict: %s", list(result.keys()))
            return None
        if "tags" not in result or not isinstance(result["tags"], list):
            logger.warning("LLM response missing 'tags' list: %s", list(result.keys()))
            return None

        # Validate scores: each category must have score (1-10) and chunk_ids list
        chunk_ids = {c["id"] for c in chunks}
        valid_scores = {}
        for cat in VALID_SCORE_CATEGORIES:
            entry = result["scores"].get(cat)
            if not isinstance(entry, dict):
                logger.warning("Score category '%s' missing or not a dict", cat)
                valid_scores[cat] = {"score": 5, "chunk_ids": []}
                continue
            score = entry.get("score")
            if not isinstance(score, int) or score < 1 or score > 10:
                logger.warning("Score '%s' has invalid score: %s", cat, score)
                score = max(1, min(10, int(score))) if isinstance(score, (int, float)) else 5
            cids = entry.get("chunk_ids", [])
            if not isinstance(cids, list):
                cids = []
            valid_cids = [cid for cid in cids if cid in chunk_ids]
            valid_scores[cat] = {"score": score, "chunk_ids": valid_cids}

        # Validate tags against actual chunk IDs and allowed types
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

        logger.info(
            "LLM analysis complete: scores=%s, tags=%d/%d valid",
            {k: v["score"] for k, v in valid_scores.items()},
            len(valid_tags), len(result["tags"]),
        )

        return {"scores": valid_scores, "tags": valid_tags}

    except json.JSONDecodeError as exc:
        logger.warning("LLM response is not valid JSON: %s", exc)
        return None
    except Exception as exc:
        logger.warning("LLM analysis failed: %s", exc)
        return None
