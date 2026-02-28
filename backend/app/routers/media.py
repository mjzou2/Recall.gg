import shutil
import time
import uuid
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import UPLOAD_DIR, DISABLE_SEMANTIC_SEARCH
from app.database import get_conn, put_conn, fetch_session, fetch_chunks
from app.services.audio import extract_audio
from app.services.embedding import embed_texts
from app.services.transcription import transcribe_audio, merge_segments


router = APIRouter(prefix="/sessions", tags=["media"])


@router.post("/{session_id}/media")
async def upload_media(session_id: str, file: UploadFile = File(...)) -> Dict:
    fetch_session(session_id)
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

    # Record start time for duration tracking
    start_time = time.time()

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET status = 'processing', audio_path = NULL WHERE id = %s",
                (session_id,),
            )
            cur.execute("DELETE FROM chunks WHERE session_id = %s", (session_id,))
        conn.commit()
    finally:
        put_conn(conn)

    try:
        print("Processing: extracting audio")
        audio_path = extract_audio(session_id, media_file)
        print("Processing: transcribing audio")
        segments = transcribe_audio(audio_path)
        print("Processing: merging segments")
        chunks = merge_segments(segments) if segments else []

        # Embed chunk texts for semantic search
        embeddings = None
        if not DISABLE_SEMANTIC_SEARCH and chunks:
            print("Processing: embedding chunks for semantic search")
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = embed_texts(chunk_texts)
            if embeddings:
                print(f"Embedded {len(embeddings)} chunks")
            else:
                print("Warning: embedding failed, chunks will have NULL embeddings")

        chunk_rows = []
        for i, chunk in enumerate(chunks):
            embedding = None
            if embeddings and i < len(embeddings):
                embedding = embeddings[i]
            chunk_rows.append((
                str(uuid.uuid4()),
                session_id,
                chunk["start_ms"],
                chunk["end_ms"],
                chunk["text"],
                chunk.get("speaker"),
                str(embedding) if embedding else None,
            ))

        # Calculate processing duration
        duration_seconds = int(time.time() - start_time)

        conn = get_conn()
        try:
            with conn.cursor() as cur:
                if chunk_rows:
                    for row in chunk_rows:
                        cur.execute(
                            """
                            INSERT INTO chunks(id, session_id, start_ms, end_ms, text, speaker, embedding)
                            VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                            """,
                            row,
                        )
                cur.execute(
                    "UPDATE sessions SET status = 'ready', audio_path = %s, processing_duration_seconds = %s WHERE id = %s",
                    (str(audio_path), duration_seconds, session_id),
                )
            conn.commit()
        finally:
            put_conn(conn)
    except Exception as exc:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET status = 'failed' WHERE id = %s",
                    (session_id,),
                )
            conn.commit()
        finally:
            put_conn(conn)
        print(f"Processing failed for session {session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Processing failed") from exc

    return {
        "session": session_id,
        "audio_path": str(audio_path),
        "chunks": fetch_chunks(session_id),
    }
