"""
Standalone diarization-tuning script.

Loads a saved 2_align.json (whisperX aligned output) and a raw audio file,
runs pyannote diarization + assign_word_speakers, then merge_segments().
Lets you tune diarization and merge parameters without re-running transcription.

Usage (from backend/ directory):
    python -m scripts.test_diarize \\
        --audio  data/audio/<session_id>.wav \\
        --align  data/uploads/<session_id>/debug/2_align.json \\
        [--num-speakers 5] \\
        [--segmentation-threshold 0.4] \\
        [--clustering-threshold 0.7] \\
        [--min-duration-on 0.2] \\
        [--min-duration-off 0.3] \\
        [--pause-threshold 2500]

Output format per chunk:
    [0:05.120 - 0:12.400] SPEAKER_00: the text goes here
"""
import argparse
import json
import sys
from pathlib import Path


def ms_to_ts(ms: int) -> str:
    total_s = ms // 1000
    frac = ms % 1000
    m, s = divmod(total_s, 60)
    return f"{m}:{s:02d}.{frac:03d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune diarization on saved 2_align.json without re-transcribing"
    )
    parser.add_argument("--audio", required=True, help="Path to audio file (.wav)")
    parser.add_argument("--align", required=True, help="Path to 2_align.json")
    parser.add_argument(
        "--num-speakers", type=int, default=None,
        help="Fix exact speaker count (sets both min and max)",
    )
    parser.add_argument(
        "--segmentation-threshold", type=float, default=None,
        help="Pyannote segmentation threshold (default: model default)",
    )
    parser.add_argument(
        "--clustering-threshold", type=float, default=None,
        help="Pyannote clustering threshold (default: model default)",
    )
    parser.add_argument(
        "--min-duration-on", type=float, default=None,
        help="Pyannote min_duration_on in seconds (default: model default)",
    )
    parser.add_argument(
        "--min-duration-off", type=float, default=None,
        help="Pyannote min_duration_off in seconds (default: model default)",
    )
    parser.add_argument(
        "--pause-threshold", type=int, default=None,
        help="Word pause threshold in ms for merge_segments (default: WORD_PAUSE_THRESHOLD from env)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write chunk output to this file in addition to stdout",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    align_path = Path(args.align)

    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)
    if not align_path.exists():
        print(f"Error: align file not found: {align_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from app.config import HF_TOKEN, WORD_PAUSE_THRESHOLD_MS
        from app.services.transcription import (
            _get_device,
            merge_segments,
        )
    except ImportError as exc:
        print(
            f"Import error: {exc}\n"
            "Run this script from the backend/ directory:\n"
            "  python -m scripts.test_diarize --audio <path> --align <path>",
            file=sys.stderr,
        )
        sys.exit(1)

    if not HF_TOKEN:
        print("Error: HF_TOKEN is not set in .env — required for diarization.", file=sys.stderr)
        sys.exit(1)

    with align_path.open() as f:
        aligned = json.load(f)

    print(f"Loaded aligned result from {align_path}")
    print(f"  segments: {len(aligned.get('segments', []))}")
    print(f"  word_segments: {len(aligned.get('word_segments', []))}")

    pause_ms = args.pause_threshold if args.pause_threshold is not None else WORD_PAUSE_THRESHOLD_MS
    device = _get_device()

    print("\nDiarization settings:")
    print(f"  device:                 {device}")
    print(f"  num_speakers:           {args.num_speakers}")
    print(f"  segmentation_threshold: {args.segmentation_threshold}")
    print(f"  clustering_threshold:   {args.clustering_threshold}")
    print(f"  min_duration_on:        {args.min_duration_on}")
    print(f"  min_duration_off:       {args.min_duration_off}")
    print(f"  pause_threshold_ms:     {pause_ms}\n")

    try:
        import whisperx
        from whisperx.diarize import DiarizationPipeline
    except ImportError as exc:
        print(f"Error: could not import whisperx: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Loading audio...")
    audio = whisperx.load_audio(str(audio_path))

    print("Loading diarization pipeline...")
    diarize_pipeline = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-community-1",
        token=HF_TOKEN,
        device=device,
    )

    # Apply pyannote hyperparameters one sub-dict at a time so a failure in one
    # (e.g. segmentation.threshold not existing in this model version) doesn't
    # silently drop the other params (clustering.threshold, min_duration_*).
    seg_params = {}
    if args.segmentation_threshold is not None:
        seg_params["threshold"] = args.segmentation_threshold
    if args.min_duration_off is not None:
        seg_params["min_duration_off"] = args.min_duration_off
    if args.min_duration_on is not None:
        seg_params["min_duration_on"] = args.min_duration_on
    if seg_params:
        try:
            diarize_pipeline.model.instantiate({"segmentation": seg_params})
            print(f"Applied segmentation params: {seg_params}")
        except Exception as exc:
            print(f"Warning: could not apply segmentation params ({exc}), using model defaults")

    if args.clustering_threshold is not None:
        try:
            diarize_pipeline.model.instantiate({"clustering": {"threshold": args.clustering_threshold}})
            print(f"Applied clustering threshold: {args.clustering_threshold}")
        except Exception as exc:
            print(f"Warning: could not apply clustering threshold ({exc}), using model default")

    diarize_kwargs = {}
    if args.num_speakers is not None:
        diarize_kwargs["min_speakers"] = args.num_speakers
        diarize_kwargs["max_speakers"] = args.num_speakers

    print("Running diarization...")
    diarize_segments = diarize_pipeline(audio, **diarize_kwargs)
    result = whisperx.assign_word_speakers(diarize_segments, aligned)
    print("Diarization complete.\n")

    # Extract word-level data from segments[i]["words"], NOT from word_segments.
    # assign_word_speakers updates the words nested inside each segment in place.
    # When aligned was loaded from JSON, word_segments is a separate list of dicts
    # (not shared references), so it never receives the speaker labels.
    words = []
    for seg in result.get("segments", []):
        for word_info in seg.get("words", []):
            word = word_info.get("word", "").strip()
            if not word:
                continue
            start = word_info.get("start")
            end = word_info.get("end")
            if start is None or end is None:
                continue
            words.append({
                "start_ms": int(start * 1000),
                "end_ms":   int(end   * 1000),
                "text":     word,
                "speaker":  word_info.get("speaker"),
            })

    print(f"Extracted {len(words)} words. Running merge_segments(pause_threshold_ms={pause_ms})...\n")
    chunks = merge_segments(words, pause_threshold_ms=pause_ms)

    lines = []
    for chunk in chunks:
        start = ms_to_ts(chunk["start_ms"])
        end = ms_to_ts(chunk["end_ms"])
        speaker = chunk.get("speaker") or "?"
        lines.append(f"[{start} - {end}] {speaker}: {chunk['text']}")

    print(f"{len(chunks)} chunks:\n")
    for line in lines:
        print(line)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            f.write(f"{len(chunks)} chunks\n\n")
            f.write("\n".join(lines) + "\n")
        print(f"\nOutput written to {out_path}")


if __name__ == "__main__":
    main()
