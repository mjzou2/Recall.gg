import shutil
import time
import uuid
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import UPLOAD_DIR, DISABLE_SEMANTIC_SEARCH, DISABLE_LLM_ANALYSIS, SAVE_INTERMEDIATE
from app.database import get_conn, put_conn, fetch_session, fetch_chunks
from app.services.audio import extract_audio
from app.services.embedding import embed_texts
from app.services.llm import analyze_session
from app.services.transcription import transcribe_audio, merge_segments


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
        debug_dir = (UPLOAD_DIR / session_id / "debug") if SAVE_INTERMEDIATE else None
        segments = transcribe_audio(audio_path, debug_dir=debug_dir)
        print("Processing: merging segments")
        chunks = merge_segments(segments) if segments else []
        if debug_dir and chunks:
            import json
            debug_dir.mkdir(parents=True, exist_ok=True)
            out_path = debug_dir / "4_chunks.json"
            with out_path.open("w") as f:
                json.dump(chunks, f, indent=2)
            print(f"[debug] saved {out_path}")

        # Embed chunk texts for semantic search
        embeddings = None
        if not DISABLE_SEMANTIC_SEARCH and chunks:
            print("Processing: embedding chunks for semantic search")
            embed_start = time.time()
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = embed_texts(chunk_texts)
            embed_elapsed = time.time() - embed_start
            if embeddings:
                print(f"Embedded {len(embeddings)} chunks in {embed_elapsed:.1f}s")
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

        # Insert chunks into database
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
            conn.commit()
        finally:
            put_conn(conn)

        # LLM analysis (fault-tolerant, after chunks committed)
        if not DISABLE_LLM_ANALYSIS and chunk_rows:
            print("Processing: running LLM analysis")
            llm_start = time.time()
            # Build chunk dicts from chunk_rows tuples
            # Format: (id, session_id, start_ms, end_ms, text, speaker, embedding)
            llm_chunks = [
                {"id": row[0], "start_ms": row[2], "end_ms": row[3],
                 "text": row[4], "speaker": row[5]}
                for row in chunk_rows
            ]
            duration_ms = max(row[3] for row in chunk_rows)  # max end_ms
            try:
                analysis = analyze_session(llm_chunks, duration_ms)
                llm_elapsed = time.time() - llm_start
                if analysis:
                    print(f"LLM analysis complete in {llm_elapsed:.1f}s: "
                          f"{len(analysis['tags'])} tags")
                    conn = get_conn()
                    try:
                        with conn.cursor() as cur:
                            # Format scorecard as session notes
                            score_labels = {
                                "summoner_tracking": "Summoner Tracking",
                                "objective_setup": "Objective Setup",
                                "teamfight_comms": "Teamfight Comms",
                                "shotcall_clarity": "Shotcall Clarity",
                                "map_awareness": "Map Awareness",
                            }
                            lines = ["SCORECARD"]
                            for key, label in score_labels.items():
                                s = analysis["scores"].get(key, {}).get("score", "?")
                                lines.append(f"{label}: {s}/10")
                            scorecard_text = "\n".join(lines)
                            cur.execute(
                                "UPDATE sessions SET notes = %s WHERE id = %s",
                                (scorecard_text, session_id),
                            )
                            # Append tags to chunk notes
                            for tag in analysis["tags"]:
                                tag_text = f"[{tag['type']}] {tag['label']}"
                                cur.execute(
                                    """UPDATE chunks SET notes =
                                       CASE WHEN notes IS NULL OR notes = '' THEN %s
                                            ELSE notes || E'\\n' || %s
                                       END
                                       WHERE id = %s""",
                                    (tag_text, tag_text, tag["chunk_id"]),
                                )
                        conn.commit()
                    finally:
                        put_conn(conn)
                else:
                    print(f"LLM analysis returned no results ({llm_elapsed:.1f}s)")
            except Exception as llm_exc:
                print(f"LLM analysis failed (non-fatal): {llm_exc}")

        # Compute session duration from chunk timestamps
        session_duration_ms = max(row[3] for row in chunk_rows) if chunk_rows else None

        # Update session status to ready
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET status = 'ready', audio_path = %s, processing_duration_seconds = %s, duration_ms = %s WHERE id = %s",
                    (str(audio_path), duration_seconds, session_duration_ms, session_id),
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
