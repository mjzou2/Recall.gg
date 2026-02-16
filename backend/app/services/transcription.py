from pathlib import Path
from typing import Dict, List, Optional

from fastapi import HTTPException
from faster_whisper import WhisperModel

from app.config import (
    WHISPER_MODEL,
    TRANSCRIBE_DEVICE,
    DISABLE_LOL_NORMALIZE,
    DISABLE_FUZZY_CORRECT,
)
from app.services.text_processing import (
    load_term_bank,
    normalize_lol_text,
    _build_fuzzy_lookup,
    fuzzy_correct_text,
)


_whisper_model: Optional[WhisperModel] = None


def get_transcriber() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        device = TRANSCRIBE_DEVICE.lower()
        is_cpu = device == "cpu"
        compute_type = "int8" if is_cpu else "float16"

        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device="cpu" if is_cpu else "cuda",
            compute_type=compute_type,
        )
    return _whisper_model


def transcribe_audio(audio_path: Path) -> List[Dict]:
    # Load term bank
    term_bank = load_term_bank()

    # Build hotwords list: all objectives + top common_comms terms
    hotwords_list = []
    if term_bank:
        # Add all objectives
        hotwords_list.extend(term_bank.get("objectives", []))
        # Add high-frequency comms terms (pick first 15)
        hotwords_list.extend(term_bank.get("common_comms", [])[:15])

    # Fallback if term bank fails to load
    if not hotwords_list:
        hotwords_list = [
            "baron", "herald", "grubs", "dragon", "drake", "ward", "reset",
            "tp", "teleport", "flash", "smite", "invade", "dive", "prio", "push"
        ]

    # Convert list to comma-separated string
    hotwords = ", ".join(hotwords_list)

    # Build initial_prompt: natural sentences with key terms
    initial_prompt = ""
    if term_bank:
        objectives = term_bank.get("objectives", [])[:5]  # First 5 objectives
        champions = term_bank.get("champions", [])[:4]    # First 4 champions
        comms = term_bank.get("common_comms", [])[:4]     # First 4 comms terms

        initial_prompt = (
            f"The team secured {objectives[0] if objectives else 'Baron'} and warded river. "
            f"{champions[0] if champions else 'The mid laner'} and {champions[1] if len(champions) > 1 else 'the jungler'} "
            f"rotated mid while tracking the enemy. They need to {comms[0] if comms else 'reset'} and get vision."
        )

    normalize_enabled = not DISABLE_LOL_NORMALIZE
    fuzzy_enabled = not DISABLE_FUZZY_CORRECT
    fuzzy_lookup = _build_fuzzy_lookup(term_bank) if fuzzy_enabled else {}
    device = TRANSCRIBE_DEVICE.lower()

    print(
        "Transcribing with model=%s device=%s hotwords=%s initial_prompt=%s normalize=%s fuzzy=%s"
        % (WHISPER_MODEL, device, bool(hotwords), bool(initial_prompt), normalize_enabled, fuzzy_enabled)
    )

    try:
        segments, _info = get_transcriber().transcribe(
            str(audio_path),
            vad_filter=True,
            hotwords=hotwords,
            initial_prompt=initial_prompt,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Transcription failed: {exc}"
        ) from exc

    results: List[Dict] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if normalize_enabled:
            text = normalize_lol_text(text)
        if fuzzy_enabled and fuzzy_lookup:
            text = fuzzy_correct_text(text, fuzzy_lookup)
        results.append(
            {
                "start_ms": int(segment.start * 1000),
                "end_ms": int(segment.end * 1000),
                "text": text,
            }
        )
    return results


def merge_segments(segments: List[Dict], max_chunk_ms: int = 35000) -> List[Dict]:
    merged: List[Dict] = []
    current: Optional[Dict] = None

    for segment in segments:
        if current is None:
            current = segment.copy()
            continue

        if segment["end_ms"] - current["start_ms"] <= max_chunk_ms:
            current["end_ms"] = segment["end_ms"]
            current["text"] = f"{current['text']} {segment['text']}"
        else:
            merged.append(current)
            current = segment.copy()

    if current is not None:
        merged.append(current)

    return merged
