import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi import HTTPException

from app.config import AUDIO_DIR, DISABLE_AUDIO_DENOISE, DISABLE_AUDIO_PREPROCESSING, SAVE_INTERMEDIATE


def extract_audio(session_id: str, media_path: Path) -> Path:
    audio_path = AUDIO_DIR / f"{session_id}.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip extraction if file is already an audio-only type.
    if media_path.suffix.lower() in {".wav", ".mp3", ".m4a", ".ogg", ".flac"}:
        return media_path

    # Light denoise filter chain for noisy Discord comms:
    #   highpass=f=200   — cut low rumble/hum below 200Hz
    #   lowpass=f=3500   — cut hiss/static above 3500Hz (voice comms don't need higher)
    #   afftdn=nf=-25    — gentle FFT-based noise reduction (lower = more aggressive)
    denoise = not DISABLE_AUDIO_DENOISE
    af_filter = "highpass=f=200,lowpass=f=3500,afftdn=nf=-25" if denoise else None

    try:
        cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(media_path),
                "-vn",
        ]
        if af_filter:
            cmd += ["-af", af_filter]
        cmd += [
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                str(audio_path),
        ]
        subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg is required to extract audio from video files.",
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Audio extraction failed: {exc.stderr.decode(errors='ignore')}",
        ) from exc

    return audio_path


def isolate_vocals(audio_path: Path, device: str, output_dir: Path) -> Path:
    """Run demucs htdemucs with --two-stems=vocals to isolate speech from game audio.

    Returns path to the vocals WAV file. Does NOT clean up output_dir — caller handles that.
    """
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",
        "-n", "htdemucs",
        "-d", device,
        "-o", str(output_dir),
        str(audio_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    # Demucs outputs to: {output_dir}/htdemucs/{stem_name}/vocals.wav
    stem_name = audio_path.stem
    vocals_path = output_dir / "htdemucs" / stem_name / "vocals.wav"
    if not vocals_path.exists():
        raise FileNotFoundError(f"Demucs did not produce expected vocals file: {vocals_path}")
    return vocals_path


def normalize_loudness(audio_path: Path, output_path: Path) -> Path:
    """Run ffmpeg loudnorm on audio to normalize volume across speakers.

    Returns output_path. Does NOT delete input — caller handles cleanup.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def preprocess_audio(audio_path: Path, device: str, debug_dir: Path | None = None) -> Path:
    """Orchestrate demucs vocal isolation + loudnorm normalization.

    Fault-tolerant: if demucs fails, logs warning and continues with raw audio.
    Cleanup: all intermediate files deleted in finally block (unless SAVE_INTERMEDIATE is on).
    """
    if DISABLE_AUDIO_PREPROCESSING:
        print("Audio preprocessing disabled, skipping demucs + loudnorm")
        return audio_path

    demucs_output_dir = Path(tempfile.mkdtemp(prefix="demucs_", dir=AUDIO_DIR))
    vocals_path: Path | None = None
    normalized_path = AUDIO_DIR / f"{audio_path.stem}_preprocessed.wav"
    current_audio = audio_path

    try:
        # Step 1: Demucs vocal isolation (fault-tolerant)
        try:
            print("Preprocessing: running demucs vocal isolation (htdemucs)")
            vocals_path = isolate_vocals(audio_path, device, demucs_output_dir)
            current_audio = vocals_path
            print(f"Demucs complete: {vocals_path}")
        except Exception as exc:
            print(f"Demucs not found or failed — skipping vocal isolation, using raw audio: {exc}")

        # Step 2: Loudnorm normalization
        print("Preprocessing: running ffmpeg loudnorm")
        normalized_path = normalize_loudness(current_audio, normalized_path)
        current_audio = normalized_path
        print(f"Loudnorm complete: {normalized_path}")

        return current_audio

    except Exception as exc:
        print(f"Audio preprocessing failed — using original audio: {exc}")
        # Clean up partial normalized output on failure
        if normalized_path.exists() and normalized_path != audio_path:
            normalized_path.unlink(missing_ok=True)
        return audio_path

    finally:
        # Save intermediates for A/B comparison if debug mode is on
        if SAVE_INTERMEDIATE and debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            if vocals_path and vocals_path.exists():
                shutil.copy2(vocals_path, debug_dir / "demucs_vocals.wav")
                print(f"[debug] saved {debug_dir / 'demucs_vocals.wav'}")
            if normalized_path.exists() and normalized_path != audio_path:
                shutil.copy2(normalized_path, debug_dir / "loudnorm_output.wav")
                print(f"[debug] saved {debug_dir / 'loudnorm_output.wav'}")

        # Clean up demucs output directory
        if demucs_output_dir.exists():
            shutil.rmtree(demucs_output_dir, ignore_errors=True)
