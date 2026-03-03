import gc
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import HTTPException

from app.config import (
    WHISPER_MODEL,
    TRANSCRIBE_DEVICE,
    WHISPER_COMPUTE_TYPE,
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
    if WHISPER_COMPUTE_TYPE:
        return WHISPER_COMPUTE_TYPE
    return "int8" if _get_device() == "cpu" else "float16"


def _unload_model(model) -> None:
    """Delete a model and free GPU memory."""
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


HOTWORDS = [
    # Apostrophe champions (Whisper mangles these consistently)
    "Kha'Zix", "Cho'Gath", "Rek'Sai", "Kai'Sa", "Bel'Veth",
    "K'Sante", "Vel'Koz", "Kog'Maw", "Jak'Sho",
    # Non-English champion names
    "Qiyana", "Naafiri", "Xayah", "Xerath", "Zyra",
    "Aphelios", "Aurelion Sol", "Renata Glasc",
    # Abbreviations & jargon
    "ADC", "BORK", "IE", "BT", "CS", "CC", "LDR", "QSS", "RFC", "ROA",
    # Esports terms not in normal English
    "prio", "crossmap", "tribush", "krugs", "gromp", "raptors",
    # Items Whisper wouldn't know
    "Zhonya's", "Liandry's", "Serylda's", "Navori", "Youmuu's",
    "Guinsoo", "Morellonomicon", "Rabadon's",
]

INITIAL_PROMPT = (
    "Okay we need prio mid before drake spawns. Kai'Sa has BORK, Rek'Sai is "
    "topside. They're setting up baron, we need to rotate. Naut no flash."
    "Zhonya's is up though. Let's crash this wave and group"
    "for elder. ADC needs to play safe, they have Kha'Zix in fog."
)


def _build_hotwords() -> str:
    """Return curated hotwords string for terms Whisper gets wrong."""
    return ", ".join(HOTWORDS)


def _build_initial_prompt() -> str:
    """Return static initial prompt that sounds like natural scrim comms."""
    return INITIAL_PROMPT


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

    hotwords = _build_hotwords()
    initial_prompt = _build_initial_prompt()

    print(
        "Transcribing with whisperX model=%s device=%s batch_size=%d normalize=%s fuzzy=%s diarize=%s"
        % (WHISPER_MODEL, device, WHISPERX_BATCH_SIZE, normalize_enabled, fuzzy_enabled, diarize_enabled)
    )

    try:
        import whisperx
        from whisperx.diarize import DiarizationPipeline

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
