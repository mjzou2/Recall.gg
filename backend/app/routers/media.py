import shutil
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import UPLOAD_DIR
from app.database import get_conn, put_conn, fetch_session


router = APIRouter(prefix="/sessions", tags=["media"])


@router.post("/{session_id}/media")
async def upload_media(session_id: str, file: UploadFile = File(...)) -> Dict:
    fetch_session(session_id)
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    MAX_SIZE = 5 * 1024 * 1024 * 1024  # 5GB in bytes
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

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET status = 'uploaded', media_path = %s, audio_path = NULL WHERE id = %s",
                (str(dest_path), session_id),
            )
        conn.commit()
    finally:
        put_conn(conn)

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

    # Reject if already queued or processing
    if session.get("status") in ("queued", "processing"):
        raise HTTPException(
            status_code=409, detail="Session is already queued or processing."
        )

    # Enqueue: set status to 'queued', clear processing_step
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET status = 'queued', processing_step = NULL WHERE id = %s",
                (session_id,),
            )
        conn.commit()
    finally:
        put_conn(conn)

    return {"session": session_id, "status": "queued"}
