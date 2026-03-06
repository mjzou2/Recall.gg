import gc
import multiprocessing
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
    DIARIZE_MIN_SPEAKERS,
    DIARIZE_MAX_SPEAKERS,
    DIARIZE_SEGMENTATION_THRESHOLD,
    DIARIZE_CLUSTERING_THRESHOLD,
    DIARIZE_MIN_DURATION_OFF,
    DIARIZE_MIN_DURATION_ON,
    WORD_PAUSE_THRESHOLD_MS,
)
from app.services.text_processing import (
    load_term_bank,
    collapse_repetitions,
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
    """Delete a model and free GPU memory, including orphaned child processes."""
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Terminate any orphaned multiprocessing workers (whisperX/pyannote spawn these
    # for GPU batch operations and don't always clean them up)
    for child in multiprocessing.active_children():
        child.terminate()
        child.join(timeout=5)


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
    "prio", "crossmap", "tribush", "krugs", "gromp", "raptors", "tempo",
    # Items Whisper wouldn't know
    "Zhonya's", "Liandry's", "Serylda's", "Navori", "Youmuu's",
    "Guinsoo", "Morellonomicon", "Rabadon's",
]


INITIAL_PROMPT = (
    "Okay we need prio mid before drake spawns. Kai'Sa has BORK, Rek'Sai is "
    "topside. They're setting up baron, we need to rotate. Naut no flash."
    "Zhonya's is up though. Let's crash this wave and group with tempo"
    "for elder. ADC needs to play safe, they have Kha'Zix in fog."
)


def _build_hotwords() -> str:
    """Return curated hotwords string for terms Whisper gets wrong."""
    return ", ".join(HOTWORDS)


def _build_initial_prompt() -> str:
    """Return static initial prompt that sounds like natural scrim comms."""
    return INITIAL_PROMPT


def _save_debug(debug_dir: Path, filename: str, data) -> None:
    """Save intermediate pipeline result as JSON for later inspection/replay."""
    import json
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / filename
    with out_path.open("w") as f:
        json.dump(data, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"[debug] saved {out_path}")


def transcribe_audio(audio_path: Path, debug_dir: Optional[Path] = None) -> List[Dict]:
    """
    Full whisperX pipeline: transcribe -> align -> diarize -> assign speakers.
    Models are loaded and unloaded sequentially to manage VRAM.
    Returns list of dicts: {start_ms, end_ms, text, speaker}.
    If debug_dir is provided, saves intermediate results as JSON after each major step.
    """
    normalize_enabled = not DISABLE_LOL_NORMALIZE
    fuzzy_enabled = not DISABLE_FUZZY_CORRECT
    diarize_enabled = bool(HF_TOKEN)

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
        if debug_dir:
            _save_debug(debug_dir, "1_transcribe.json", result)

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
        if debug_dir:
            _save_debug(debug_dir, "2_align.json", result)

        # Step 4: Diarize (optional, fault-tolerant)
        if diarize_enabled:
            try:
                diarize_pipeline = DiarizationPipeline(
                    model_name="pyannote/speaker-diarization-community-1",
                    token=HF_TOKEN,
                    device=device,
                )
                # Speaker count: DIARIZE_MIN/MAX_SPEAKERS take priority; NUM_SPEAKERS is fallback
                min_spk = DIARIZE_MIN_SPEAKERS if DIARIZE_MIN_SPEAKERS is not None else NUM_SPEAKERS
                max_spk = DIARIZE_MAX_SPEAKERS if DIARIZE_MAX_SPEAKERS is not None else NUM_SPEAKERS
                diarize_kwargs: Dict = {}
                if min_spk is not None:
                    diarize_kwargs["min_speakers"] = min_spk
                if max_spk is not None:
                    diarize_kwargs["max_speakers"] = max_spk

                # Apply pyannote hyperparameters if any threshold vars are set
                hyper: Dict = {}
                if DIARIZE_SEGMENTATION_THRESHOLD is not None or DIARIZE_MIN_DURATION_OFF is not None or DIARIZE_MIN_DURATION_ON is not None:
                    seg_params: Dict = {}
                    if DIARIZE_SEGMENTATION_THRESHOLD is not None:
                        seg_params["threshold"] = DIARIZE_SEGMENTATION_THRESHOLD
                    if DIARIZE_MIN_DURATION_OFF is not None:
                        seg_params["min_duration_off"] = DIARIZE_MIN_DURATION_OFF
                    if DIARIZE_MIN_DURATION_ON is not None:
                        seg_params["min_duration_on"] = DIARIZE_MIN_DURATION_ON
                    hyper["segmentation"] = seg_params
                if DIARIZE_CLUSTERING_THRESHOLD is not None:
                    hyper["clustering"] = {"threshold": DIARIZE_CLUSTERING_THRESHOLD}
                if hyper:
                    try:
                        diarize_pipeline.model.instantiate(hyper)
                        print(f"Diarization hyperparameters applied: {hyper}")
                    except Exception as hyper_exc:
                        print(f"Could not apply diarization hyperparameters ({hyper_exc}), using model defaults")

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

    # Step 5: Extract word-level data (speaker assigned per-word by assign_word_speakers)
    results: List[Dict] = []
    for word_info in result.get("word_segments", []):
        word = word_info.get("word", "").strip()
        if not word:
            continue
        start = word_info.get("start")
        end = word_info.get("end")
        if start is None or end is None:
            continue
        speaker = word_info.get("speaker") if diarize_enabled else None
        results.append(
            {
                "start_ms": int(start * 1000),
                "end_ms":   int(end   * 1000),
                "text":     word,
                "speaker":  speaker,
            }
        )
    if debug_dir:
        _save_debug(debug_dir, "3_diarize.json", results)
    return results


def _smooth_speaker_labels(words: List[Dict]) -> List[Dict]:
    """
    Remove isolated single-word speaker misassignments caused by noisy word-level diarization.

    If word[i].speaker differs from both word[i-1].speaker and word[i+1].speaker,
    reassign it to match the previous word's speaker.  Sequences of 2+ consecutive
    words with the same speaker are left untouched — those are likely real speaker
    changes.  Only operates when all three adjacent speakers are known (non-None).
    """
    if len(words) < 3:
        return words

    smoothed = [w.copy() for w in words]
    for i in range(1, len(smoothed) - 1):
        prev_spk = smoothed[i - 1].get("speaker")
        curr_spk = smoothed[i].get("speaker")
        next_spk = smoothed[i + 1].get("speaker")
        if prev_spk is None or curr_spk is None or next_spk is None:
            continue
        if curr_spk != prev_spk and curr_spk != next_spk:
            smoothed[i]["speaker"] = prev_spk
    return smoothed


def _fix_trailing_words(
    chunks: List[Dict],
    max_word_duration_ms: int = 300,
    max_gap_ms: int = 200,
) -> List[Dict]:
    """
    Correct trailing-word diarization errors at speaker boundaries.

    At each cross-speaker boundary, inspect the last word of the earlier chunk.
    If BOTH conditions hold:
      - word duration < max_word_duration_ms  (short/clipped → likely cut off)
      - gap between the two chunks < max_gap_ms  (nearly touching → no real pause)
    then move that word to the front of the next chunk and update timestamps.
    If the move empties the earlier chunk, delete it.

    Requires chunks to have a temporary '_words' list (set by merge_segments).
    Operates in a single forward pass; mutates and returns the chunk list.
    """
    i = 0
    while i < len(chunks) - 1:
        a = chunks[i]
        b = chunks[i + 1]

        spk_a = a.get("speaker")
        spk_b = b.get("speaker")
        # Only act at real cross-speaker boundaries where both labels are known
        if spk_a is None or spk_b is None or spk_a == spk_b:
            i += 1
            continue

        words_a = a.get("_words", [])
        if not words_a:
            i += 1
            continue

        last = words_a[-1]
        word_duration = last["end_ms"] - last["start_ms"]
        gap = b["start_ms"] - a["end_ms"]

        if word_duration < max_word_duration_ms and gap < max_gap_ms:
            # Move last word of a → front of b
            words_a.pop()
            words_b = b.setdefault("_words", [])
            words_b.insert(0, last)

            # Rebuild b from its updated word list
            b["start_ms"] = words_b[0]["start_ms"]
            b["text"] = " ".join(w["text"] for w in words_b)

            if words_a:
                # Shrink a
                a["end_ms"] = words_a[-1]["end_ms"]
                a["text"] = " ".join(w["text"] for w in words_a)
                i += 1
            else:
                # a is now empty — remove it; don't advance i (new chunks[i] is old b)
                chunks.pop(i)
        else:
            i += 1

    return chunks


def merge_segments(
    words: List[Dict],
    pause_threshold_ms: int = WORD_PAUSE_THRESHOLD_MS,
) -> List[Dict]:
    """
    Build chunks from word-level dicts (output of transcribe_audio).
    Split rules (in priority order):
      1. Speaker change → always split, regardless of chunk duration
      2. Pause gap > pause_threshold_ms between consecutive words → split
      3. Otherwise → append word to current chunk
    No minimum chunk duration — one-word chunks are fine.
    Text normalization is applied to each chunk after it is finalized.
    """
    # Smooth out isolated single-word speaker misassignments before building chunks.
    # Word-level diarization is noisy; a single word labelled with a different speaker
    # than both its neighbours is almost certainly a mis-assignment, not a real turn.
    words = _smooth_speaker_labels(words)

    normalize_enabled = not DISABLE_LOL_NORMALIZE
    fuzzy_enabled = not DISABLE_FUZZY_CORRECT
    term_bank = load_term_bank()
    fuzzy_lookup = _build_fuzzy_lookup(term_bank) if fuzzy_enabled else {}

    chunks: List[Dict] = []
    current: Optional[Dict] = None

    for word in words:
        if current is None:
            current = word.copy()
            current["_words"] = [{"text": word["text"], "start_ms": word["start_ms"], "end_ms": word["end_ms"]}]
            continue

        # Rule 1: speaker change (only when both labels are known)
        speaker_changed = (
            word.get("speaker") is not None
            and current.get("speaker") is not None
            and word["speaker"] != current["speaker"]
        )
        # Rule 2: pause gap between consecutive words
        gap_ms = word["start_ms"] - current["end_ms"]
        pause_exceeded = gap_ms > pause_threshold_ms

        if speaker_changed or pause_exceeded:
            chunks.append(current)
            current = word.copy()
            current["_words"] = [{"text": word["text"], "start_ms": word["start_ms"], "end_ms": word["end_ms"]}]
        else:
            current["end_ms"] = word["end_ms"]
            current["text"] = f"{current['text']} {word['text']}"
            current["_words"].append({"text": word["text"], "start_ms": word["start_ms"], "end_ms": word["end_ms"]})

    if current is not None:
        chunks.append(current)

    # Fix trailing-word diarization errors at speaker boundaries
    _fix_trailing_words(chunks)

    # Strip internal word list before returning
    for chunk in chunks:
        chunk.pop("_words", None)

    # Apply text normalization to each finalized chunk
    for chunk in chunks:
        text = chunk["text"]
        text = collapse_repetitions(text)
        if normalize_enabled:
            text = normalize_lol_text(text)
        if fuzzy_enabled and fuzzy_lookup:
            text = fuzzy_correct_text(text, fuzzy_lookup)
        chunk["text"] = text

    return chunks
