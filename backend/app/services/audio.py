import os
import subprocess
from pathlib import Path

from fastapi import HTTPException

from app.config import AUDIO_DIR, DISABLE_AUDIO_DENOISE


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
