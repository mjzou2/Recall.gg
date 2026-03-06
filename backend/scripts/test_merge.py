"""
Standalone merge-tuning script.

Usage:
    python -m scripts.test_merge path/to/3_diarize.json [--pause-threshold 2500]

Loads the word-level diarized output saved by SAVE_INTERMEDIATE=1 and reruns
merge_segments() so you can tune split logic without waiting for transcription.

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
    parser = argparse.ArgumentParser(description="Test merge_segments on saved 3_diarize.json")
    parser.add_argument("json_path", help="Path to 3_diarize.json")
    parser.add_argument(
        "--pause-threshold",
        type=int,
        default=None,
        help="Pause threshold in ms (default: WORD_PAUSE_THRESHOLD from env / 1500)",
    )
    parser.add_argument(
        "--pause-ms",
        type=int,
        default=None,
        help="Alias for --pause-threshold (deprecated)",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    with json_path.open() as f:
        words = json.load(f)

    print(f"Loaded {len(words)} words from {json_path}")

    # Import merge_segments from the backend package (run from backend/ dir)
    try:
        from app.services.transcription import merge_segments, WORD_PAUSE_THRESHOLD_MS
    except ImportError as exc:
        print(
            f"Import error: {exc}\n"
            "Run this script from the backend/ directory:\n"
            "  python -m scripts.test_merge path/to/3_diarize.json",
            file=sys.stderr,
        )
        sys.exit(1)

    pause_ms = args.pause_threshold if args.pause_threshold is not None else (
        args.pause_ms if args.pause_ms is not None else WORD_PAUSE_THRESHOLD_MS
    )
    print(f"pause_threshold_ms={pause_ms}\n")

    chunks = merge_segments(words, pause_threshold_ms=pause_ms)

    print(f"{len(chunks)} chunks:\n")
    for chunk in chunks:
        start = ms_to_ts(chunk["start_ms"])
        end = ms_to_ts(chunk["end_ms"])
        speaker = chunk.get("speaker") or "?"
        print(f"[{start} - {end}] {speaker}: {chunk['text']}")


if __name__ == "__main__":
    main()
