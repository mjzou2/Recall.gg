"""
Experimental "diarize first, transcribe per speaker" pipeline.

Instead of the standard pipeline (transcribe → align → diarize → assign speakers),
this script:
  1. Diarizes raw audio to get speaker time regions
  2. Slices audio per speaker into temporary wav files
  3. Transcribes each speaker's audio independently with whisperX
  4. Reassembles words back to original timestamps
  5. Merges into chronological chunks via merge_segments()

The hypothesis: transcribing isolated single-speaker audio should give Whisper
cleaner input (no overlapping voices) and potentially better accuracy.

Usage (from backend/ directory):
    python -m scripts.test_diarize_first \
        --audio data/audio/<session_id>.wav \
        [--num-speakers 5] \
        [--model base.en] \
        [--device cpu] \
        [--batch-size 16] \
        [--output results.txt]
"""
# Preload pip-installed cuDNN before whisperX/ctranslate2 loads its own.
# Without this, ctranslate2 finds version-mismatched system cuDNN and crashes.
# (Same fix as backend/main.py)
import ctypes
import glob as _glob
import os as _os

try:
    import nvidia.cudnn  # type: ignore[import-untyped]
    _cudnn_dir = _os.path.join(nvidia.cudnn.__path__[0], "lib")
    for _lib in sorted(_glob.glob(_os.path.join(_cudnn_dir, "libcudnn*.so.9"))):
        ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
except (ImportError, OSError, IndexError):
    pass

import argparse
import gc
import multiprocessing
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def ms_to_ts(ms: int) -> str:
    total_s = ms // 1000
    frac = ms % 1000
    m, s = divmod(total_s, 60)
    return f"{m}:{s:02d}.{frac:03d}"


def _unload_model(model) -> None:
    """Delete a model and free GPU memory."""
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for child in multiprocessing.active_children():
        child.terminate()
        child.join(timeout=5)


def step1_diarize(
    audio_path: Path,
    device: str,
    hf_token: str,
    num_speakers: Optional[int],
) -> List[Dict]:
    """
    Run pyannote diarization on raw audio.
    Returns list of {speaker, start_s, end_s} segments.
    """
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    audio = whisperx.load_audio(str(audio_path))

    diarize_pipeline = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-community-1",
        token=hf_token,
        device=device,
    )

    diarize_kwargs: Dict = {}
    if num_speakers is not None:
        diarize_kwargs["min_speakers"] = num_speakers
        diarize_kwargs["max_speakers"] = num_speakers

    diarize_result = diarize_pipeline(audio, **diarize_kwargs)
    _unload_model(diarize_pipeline)

    # diarize_result is a pd.DataFrame with columns: start, end, speaker
    # (from whisperx's DiarizationPipeline)
    segments = []
    for _, row in diarize_result.iterrows():
        segments.append({
            "speaker": row["speaker"] if "speaker" in row.index else str(row.iloc[2]),
            "start_s": float(row["start"] if "start" in row.index else row.iloc[0]),
            "end_s": float(row["end"] if "end" in row.index else row.iloc[1]),
        })

    # Sort by start time
    segments.sort(key=lambda s: s["start_s"])
    return segments


def step2_slice_audio(
    audio_path: Path,
    diarize_segments: List[Dict],
    tmp_dir: str,
) -> Dict[str, Tuple[Path, List[Dict]]]:
    """
    For each speaker, extract and concatenate their audio segments.

    Returns dict: {
        speaker_label: (
            wav_path,
            [{"original_start_s", "original_end_s", "slice_offset_s"}, ...]
        )
    }
    """
    from pydub import AudioSegment

    full_audio = AudioSegment.from_file(str(audio_path))

    # Group segments by speaker
    speaker_segments: Dict[str, List[Dict]] = {}
    for seg in diarize_segments:
        spk = seg["speaker"]
        speaker_segments.setdefault(spk, []).append(seg)

    result: Dict[str, Tuple[Path, List[Dict]]] = {}

    for speaker, segs in sorted(speaker_segments.items()):
        combined = AudioSegment.empty()
        mapping: List[Dict] = []
        offset_ms = 0

        for seg in segs:
            start_ms = int(seg["start_s"] * 1000)
            end_ms = int(seg["end_s"] * 1000)
            # Clamp to audio length
            start_ms = max(0, start_ms)
            end_ms = min(len(full_audio), end_ms)
            if end_ms <= start_ms:
                continue

            slice_audio = full_audio[start_ms:end_ms]
            combined += slice_audio

            mapping.append({
                "original_start_s": seg["start_s"],
                "original_end_s": seg["end_s"],
                "slice_offset_s": offset_ms / 1000.0,
                "slice_duration_s": (end_ms - start_ms) / 1000.0,
            })
            offset_ms += (end_ms - start_ms)

        if len(combined) == 0:
            continue

        # Export to temp wav
        wav_path = Path(tmp_dir) / f"{speaker}.wav"
        combined.export(str(wav_path), format="wav")
        result[speaker] = (wav_path, mapping)

    return result


def step3_load_models(
    model_name: str,
    device: str,
    compute_type: str,
    hotwords: str,
    initial_prompt: str,
) -> Tuple:
    """Load whisper and align models once. Returns (whisper_model, align_model, align_metadata)."""
    import whisperx

    asr_options = {"hotwords": hotwords, "initial_prompt": initial_prompt}
    whisper_model = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language="en",
        asr_options=asr_options,
    )
    align_model, align_metadata = whisperx.load_align_model(
        language_code="en", device=device,
    )
    return whisper_model, align_model, align_metadata


def step3_transcribe_speaker(
    wav_path: Path,
    whisper_model,
    align_model,
    align_metadata,
    device: str,
    batch_size: int,
) -> List[Dict]:
    """
    Transcribe a single speaker's concatenated audio using pre-loaded models.
    Returns list of {start_s, end_s, text} word-level results (in slice-local time).
    """
    import whisperx

    audio = whisperx.load_audio(str(wav_path))

    # Transcribe
    result = whisper_model.transcribe(audio, batch_size=batch_size)

    # Align for word-level timestamps
    result = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Extract words
    words = []
    for word_info in result.get("word_segments", []):
        word = word_info.get("word", "").strip()
        if not word:
            continue
        start = word_info.get("start")
        end = word_info.get("end")
        if start is None or end is None:
            continue
        words.append({
            "start_s": float(start),
            "end_s": float(end),
            "text": word,
        })

    return words


def step4_reassemble(
    speaker_results: Dict[str, Tuple[List[Dict], List[Dict]]],
) -> List[Dict]:
    """
    Map transcribed words back to original timestamps using slice offset mapping.

    speaker_results: {speaker: (words_in_slice_time, mapping)}
    Returns: list of {start_ms, end_ms, text, speaker} sorted chronologically.
    """
    all_words: List[Dict] = []

    for speaker, (words, mapping) in speaker_results.items():
        for word in words:
            slice_time_s = word["start_s"]

            # Find which mapping segment this word falls into
            original_start = _map_to_original(slice_time_s, mapping)
            original_end = _map_to_original(word["end_s"], mapping)

            if original_start is None or original_end is None:
                continue

            all_words.append({
                "start_ms": int(original_start * 1000),
                "end_ms": int(original_end * 1000),
                "text": word["text"],
                "speaker": speaker,
            })

    # Sort chronologically
    all_words.sort(key=lambda w: w["start_ms"])
    return all_words


def _map_to_original(
    slice_time_s: float,
    mapping: List[Dict],
) -> Optional[float]:
    """
    Convert a time position in the concatenated slice audio back to original audio time.
    Walks through the mapping segments to find which one contains slice_time_s.
    """
    cumulative_offset = 0.0
    for m in mapping:
        seg_end_in_slice = cumulative_offset + m["slice_duration_s"]
        if slice_time_s <= seg_end_in_slice + 0.01:  # small tolerance
            # How far into this segment
            offset_within = slice_time_s - cumulative_offset
            # Clamp to segment duration
            offset_within = min(offset_within, m["slice_duration_s"])
            return m["original_start_s"] + offset_within
        cumulative_offset = seg_end_in_slice

    # Past the end — map to end of last segment
    if mapping:
        return mapping[-1]["original_end_s"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimental: diarize first, then transcribe per speaker"
    )
    parser.add_argument("--audio", required=True, help="Path to audio file (.wav)")
    parser.add_argument("--num-speakers", type=int, default=5, help="Number of speakers (default: 5)")
    parser.add_argument("--model", default="base.en", help="Whisper model (default: base.en)")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=16, help="WhisperX batch size (default: 16)")
    parser.add_argument("--output", default=None, help="Write output to file")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # Load .env for HF_TOKEN
    try:
        from app.config import HF_TOKEN, WHISPER_COMPUTE_TYPE
    except ImportError as exc:
        print(
            f"Import error: {exc}\n"
            "Run this script from the backend/ directory:\n"
            "  python -m scripts.test_diarize_first --audio <path>",
            file=sys.stderr,
        )
        sys.exit(1)

    if not HF_TOKEN:
        print("Error: HF_TOKEN is not set in .env — required for diarization.", file=sys.stderr)
        sys.exit(1)

    compute_type = args.device
    if WHISPER_COMPUTE_TYPE:
        compute_type = WHISPER_COMPUTE_TYPE
    elif args.device == "cpu":
        compute_type = "int8"
    else:
        compute_type = "float16"

    # Get hotwords and initial prompt from transcription module
    from app.services.transcription import _build_hotwords, _build_initial_prompt
    hotwords = _build_hotwords()
    initial_prompt = _build_initial_prompt()

    print("=" * 60)
    print("EXPERIMENTAL: Diarize-first pipeline")
    print("=" * 60)
    print(f"  audio:        {audio_path}")
    print(f"  num_speakers: {args.num_speakers}")
    print(f"  model:        {args.model}")
    print(f"  device:       {args.device}")
    print(f"  compute_type: {compute_type}")
    print(f"  batch_size:   {args.batch_size}")
    print()

    total_start = time.time()

    # --- Step 1: Diarize ---
    print("Step 1: Diarizing raw audio...")
    t0 = time.time()
    diarize_segments = step1_diarize(
        audio_path, args.device, HF_TOKEN, args.num_speakers,
    )
    t1 = time.time()
    print(f"  Found {len(diarize_segments)} diarization segments")
    speakers = sorted(set(s["speaker"] for s in diarize_segments))
    print(f"  Speakers: {speakers}")
    total_duration = max(s["end_s"] for s in diarize_segments) if diarize_segments else 0
    print(f"  Audio duration: {total_duration:.1f}s")
    for spk in speakers:
        spk_segs = [s for s in diarize_segments if s["speaker"] == spk]
        spk_dur = sum(s["end_s"] - s["start_s"] for s in spk_segs)
        print(f"    {spk}: {len(spk_segs)} segments, {spk_dur:.1f}s total")
    print(f"  Step 1 time: {t1 - t0:.1f}s\n")

    # --- Step 2: Slice audio per speaker ---
    print("Step 2: Slicing audio per speaker...")
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="diarize_first_") as tmp_dir:
        speaker_audio = step2_slice_audio(audio_path, diarize_segments, tmp_dir)
        t1 = time.time()
        for spk, (wav_path, mapping) in sorted(speaker_audio.items()):
            total_dur = sum(m["slice_duration_s"] for m in mapping)
            print(f"  {spk}: {len(mapping)} slices, {total_dur:.1f}s → {wav_path.name}")
        print(f"  Step 2 time: {t1 - t0:.1f}s\n")

        # --- Step 3: Transcribe each speaker ---
        print("Step 3a: Loading whisper + align models (once)...")
        t0 = time.time()
        whisper_model, align_model, align_metadata = step3_load_models(
            args.model, args.device, compute_type, hotwords, initial_prompt,
        )
        t_load = time.time()
        print(f"  Models loaded in {t_load - t0:.1f}s")

        print("Step 3b: Transcribing each speaker...")
        speaker_results: Dict[str, Tuple[List[Dict], List[Dict]]] = {}

        for spk, (wav_path, mapping) in sorted(speaker_audio.items()):
            spk_dur = sum(m["slice_duration_s"] for m in mapping)
            print(f"  Transcribing {spk} ({spk_dur:.1f}s)...")
            spk_t0 = time.time()
            words = step3_transcribe_speaker(
                wav_path,
                whisper_model,
                align_model,
                align_metadata,
                args.device,
                args.batch_size,
            )
            spk_t1 = time.time()
            print(f"    → {len(words)} words in {spk_t1 - spk_t0:.1f}s")
            speaker_results[spk] = (words, mapping)

        _unload_model(whisper_model)
        _unload_model(align_model)

        t1 = time.time()
        print(f"  Step 3 total time: {t1 - t0:.1f}s (load: {t_load - t0:.1f}s, transcribe: {t1 - t_load:.1f}s)\n")

    # --- Step 4: Reassemble ---
    print("Step 4: Reassembling words to original timestamps...")
    t0 = time.time()
    all_words = step4_reassemble(speaker_results)
    print(f"  Total words: {len(all_words)}")
    t1 = time.time()
    print(f"  Step 4 time: {t1 - t0:.1f}s\n")

    # --- Merge into chunks ---
    print("Step 4b: Merging into chunks via merge_segments()...")
    t0 = time.time()
    from app.services.transcription import merge_segments
    chunks = merge_segments(all_words)
    t1 = time.time()
    print(f"  {len(chunks)} chunks")
    print(f"  Step 4b time: {t1 - t0:.1f}s\n")

    total_end = time.time()

    # --- Step 5: Print results ---
    print("=" * 60)
    print(f"RESULTS: {len(chunks)} chunks")
    print(f"Total pipeline time: {total_end - total_start:.1f}s")
    print("=" * 60)
    print()

    lines = []
    for chunk in chunks:
        start = ms_to_ts(chunk["start_ms"])
        end = ms_to_ts(chunk["end_ms"])
        speaker = chunk.get("speaker") or "?"
        lines.append(f"[{start} - {end}] {speaker}: {chunk['text']}")

    for line in lines:
        print(line)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            f.write(f"EXPERIMENTAL: Diarize-first pipeline\n")
            f.write(f"Audio: {audio_path}\n")
            f.write(f"Model: {args.model}, Device: {args.device}\n")
            f.write(f"Speakers: {args.num_speakers}\n")
            f.write(f"Total time: {total_end - total_start:.1f}s\n")
            f.write(f"{len(chunks)} chunks\n\n")
            f.write("\n".join(lines) + "\n")
        print(f"\nOutput written to {out_path}")


if __name__ == "__main__":
    main()
