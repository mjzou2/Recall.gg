import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import DB_PATH, UPLOAD_DIR
from app.database import fetch_session, fetch_chunks
from app.services.audio import extract_audio
from app.services.transcription import transcribe_audio, merge_segments


router = APIRouter(prefix="/sessions", tags=["media"])


@router.post("/{session_id}/media")
async def upload_media(session_id: str, file: UploadFile = File(...)) -> Dict:
    session = fetch_session(session_id)
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    MAX_SIZE = 5 * 1024 * 1024 * 1024 # 100MB in bytes
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Seek back to start

    if file_size > MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size is 5GB, got {file_size / 1024 / 1024 / 1024:.1f}GB"
        )

    dest_dir = UPLOAD_DIR / session_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(file.filename).name

    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE sessions SET status = 'uploaded', media_path = ?, audio_path = NULL WHERE id = ?",
            (str(dest_path), session_id),
        )

    return {
        "session": session_id,
        "stored_path": str(dest_path),
        "original_filename": file.filename,
    }


@router.post("/{session_id}/process")
def process_media(session_id: str) -> Dict:
    session = fetch_session(session_id)
    media_path = session.get("media_path")
    if not media_path:
        raise HTTPException(
            status_code=400, detail="Upload a media file before processing."
        )

    media_file = Path(media_path)
    if not media_file.exists():
        raise HTTPException(status_code=400, detail="Stored media file is missing.")

    # Record start time for duration tracking
    start_time = time.time()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE sessions SET status = 'processing', audio_path = NULL WHERE id = ?",
            (session_id,),
        )
        conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))

    try:
        print("Processing: extracting audio")
        audio_path = extract_audio(session_id, media_file)
        print("Processing: transcribing audio")
        segments = transcribe_audio(audio_path)
        print("Processing: merging segments")
        chunks = merge_segments(segments) if segments else []

        chunk_rows = [
            (
                str(uuid.uuid4()),
                session_id,
                chunk["start_ms"],
                chunk["end_ms"],
                chunk["text"],
            )
            for chunk in chunks
        ]

        # Calculate processing duration
        duration_seconds = int(time.time() - start_time)

        with sqlite3.connect(DB_PATH) as conn:
            if chunk_rows:
                conn.executemany(
                    """
                    INSERT INTO chunks(id, session_id, start_ms, end_ms, text)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    chunk_rows,
                )
            conn.execute(
                "UPDATE sessions SET status = 'ready', audio_path = ?, processing_duration_seconds = ? WHERE id = ?",
                (str(audio_path), duration_seconds, session_id),
            )
    except Exception as exc:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "UPDATE sessions SET status = 'failed' WHERE id = ?",
                (session_id,),
            )
        print(f"Processing failed for session {session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Processing failed") from exc

    return {
        "session": session_id,
        "audio_path": str(audio_path),
        "chunks": fetch_chunks(session_id),
    }
