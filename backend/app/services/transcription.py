import gc
from pathlib import Path
from typing import Dict, List, Optional

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from fastapi import HTTPException

from app.config import (
    WHISPER_MODEL,
    TRANSCRIBE_DEVICE,
    DISABLE_LOL_NORMALIZE,
    DISABLE_FUZZY_CORRECT,
    HF_TOKEN,
    NUM_SPEAKERS,
    WHISPERX_BATCH_SIZE,
    MIN_CHUNK_MS,
)
from app.services.text_processing import (
    load_term_bank,
    normalize_lol_text,
    _build_fuzzy_lookup,
    fuzzy_correct_text,
)


def _get_device() -> str:
    device = TRANSCRIBE_DEVICE.lower()
    return "cpu" if device == "cpu" else "cuda"


def _get_compute_type() -> str:
    return "int8" if _get_device() == "cpu" else "float16"


def _unload_model(model) -> None:
    """Delete a model and free GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_hotwords(term_bank: Dict) -> str:
    """Build hotwords string from term bank (max ~35 terms)."""
    hotwords_list: List[str] = []
    if term_bank:
        hotwords_list.extend(term_bank.get("objectives", []))
        hotwords_list.extend(term_bank.get("common_comms", [])[:15])

    if not hotwords_list:
        hotwords_list = [
            "baron", "herald", "grubs", "dragon", "drake", "ward", "reset",
            "tp", "teleport", "flash", "smite", "invade", "dive", "prio", "push"
        ]

    return ", ".join(hotwords_list)


def _build_initial_prompt(term_bank: Dict) -> str:
    """Build initial prompt with key terms for Whisper context."""
    if not term_bank:
        return ""

    objectives = term_bank.get("objectives", [])[:5]
    champions = term_bank.get("champions", [])[:4]
    comms = term_bank.get("common_comms", [])[:4]

    return (
        f"The team secured {objectives[0] if objectives else 'Baron'} and warded river. "
        f"{champions[0] if champions else 'The mid laner'} and "
        f"{champions[1] if len(champions) > 1 else 'the jungler'} "
        f"rotated mid while tracking the enemy. They need to "
        f"{comms[0] if comms else 'reset'} and get vision."
    )


def transcribe_audio(audio_path: Path) -> List[Dict]:
    """
    Full whisperX pipeline: transcribe -> align -> diarize -> assign speakers.
    Models are loaded and unloaded sequentially to manage VRAM.
    Returns list of dicts: {start_ms, end_ms, text, speaker}.
    """
    normalize_enabled = not DISABLE_LOL_NORMALIZE
    fuzzy_enabled = not DISABLE_FUZZY_CORRECT
    diarize_enabled = bool(HF_TOKEN)

    term_bank = load_term_bank()
    fuzzy_lookup = _build_fuzzy_lookup(term_bank) if fuzzy_enabled else {}
    device = _get_device()
    compute_type = _get_compute_type()

    hotwords = _build_hotwords(term_bank)
    initial_prompt = _build_initial_prompt(term_bank)

    print(
        "Transcribing with whisperX model=%s device=%s batch_size=%d normalize=%s fuzzy=%s diarize=%s"
        % (WHISPER_MODEL, device, WHISPERX_BATCH_SIZE, normalize_enabled, fuzzy_enabled, diarize_enabled)
    )

    try:
        # Step 1: Load audio
        audio = whisperx.load_audio(str(audio_path))

        # Step 2: Transcribe
        asr_options = {"hotwords": hotwords, "initial_prompt": initial_prompt}
        model = whisperx.load_model(
            WHISPER_MODEL,
            device=device,
            compute_type=compute_type,
            language="en",
            asr_options=asr_options,
        )
        result = model.transcribe(audio, batch_size=WHISPERX_BATCH_SIZE)
        print(f"Transcription complete: {len(result.get('segments', []))} segments")
        _unload_model(model)

        # Step 3: Align (word-level timestamps)
        align_model, align_metadata = whisperx.load_align_model(
            language_code="en", device=device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        print(f"Alignment complete: {len(result.get('segments', []))} segments")
        _unload_model(align_model)

        # Step 4: Diarize (optional, fault-tolerant)
        if diarize_enabled:
            try:
                diarize_pipeline = DiarizationPipeline(
                    model_name="pyannote/speaker-diarization-community-1",
                    token=HF_TOKEN,
                    device=device,
                )
                diarize_kwargs: Dict = {}
                if NUM_SPEAKERS is not None:
                    diarize_kwargs["min_speakers"] = NUM_SPEAKERS
                    diarize_kwargs["max_speakers"] = NUM_SPEAKERS
                diarize_segments = diarize_pipeline(audio, **diarize_kwargs)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                print("Diarization complete")
                _unload_model(diarize_pipeline)
            except Exception as diarize_exc:
                print(f"Diarization failed, continuing without speaker labels: {diarize_exc}")

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Transcription failed: {exc}"
        ) from exc

    # Step 5: Extract segments with text processing
    results: List[Dict] = []
    for segment in result.get("segments", []):
        text = segment.get("text", "").strip()
        if not text:
            continue
        if normalize_enabled:
            text = normalize_lol_text(text)
        if fuzzy_enabled and fuzzy_lookup:
            text = fuzzy_correct_text(text, fuzzy_lookup)

        speaker = segment.get("speaker") if diarize_enabled else None

        results.append(
            {
                "start_ms": int(segment["start"] * 1000),
                "end_ms": int(segment["end"] * 1000),
                "text": text,
                "speaker": speaker,
            }
        )
    return results


def merge_segments(
    segments: List[Dict],
    max_chunk_ms: int = 35000,
    min_chunk_ms: int = MIN_CHUNK_MS,
) -> List[Dict]:
    """
    Merge consecutive segments into chunks.
    Breaks on EITHER:
      1. Speaker change (only if current chunk >= min_chunk_ms)
      2. Time exceeds max_chunk_ms
    Each merged chunk inherits the speaker from its first segment.
    """
    merged: List[Dict] = []
    current: Optional[Dict] = None

    for segment in segments:
        if current is None:
            current = segment.copy()
            continue

        same_speaker = (
            current.get("speaker") is None
            or segment.get("speaker") is None
            or current.get("speaker") == segment.get("speaker")
        )
        within_time = segment["end_ms"] - current["start_ms"] <= max_chunk_ms
        chunk_duration = current["end_ms"] - current["start_ms"]
        meets_minimum = chunk_duration >= min_chunk_ms

        if within_time and (same_speaker or not meets_minimum):
            current["end_ms"] = segment["end_ms"]
            current["text"] = f"{current['text']} {segment['text']}"
        else:
            merged.append(current)
            current = segment.copy()

    if current is not None:
        merged.append(current)

    return merged
