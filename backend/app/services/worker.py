import json
import threading
import time
import uuid
from pathlib import Path

import psycopg2.extras

from app.config import (
    UPLOAD_DIR,
    DISABLE_SEMANTIC_SEARCH,
    DISABLE_LLM_ANALYSIS,
    SAVE_INTERMEDIATE,
    TRANSCRIBE_DEVICE,
)
from app.database import get_conn, put_conn
from app.services.audio import extract_audio, preprocess_audio
from app.services.embedding import embed_texts
from app.services.llm import analyze_session
from app.services.transcription import transcribe_audio, merge_segments

POLL_INTERVAL = 2  # seconds

STEP_ORDER = {
    "extracting_audio": 1,
    "preprocessing": 2,
    "transcribing": 3,
    "merging": 4,
    "embedding": 5,
    "analyzing": 6,
}
TOTAL_STEPS = len(STEP_ORDER)


def recovery_on_startup() -> None:
    """Mark any sessions left in 'processing' or 'queued' as 'failed'.

    Catches jobs that were interrupted by a server crash.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET status = 'failed', processing_step = NULL "
                "WHERE status IN ('processing', 'queued')"
            )
        conn.commit()
    finally:
        put_conn(conn)


def _update_step(session_id: str, step: str) -> None:
    """Update the processing_step column for progress tracking."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET processing_step = %s WHERE id = %s",
                (step, session_id),
            )
        conn.commit()
    finally:
        put_conn(conn)


def _process_session(session_id: str) -> None:
    """Run the full processing pipeline for a queued session."""
    start_time = time.time()

    # Set status to 'processing', clear audio_path, delete existing chunks
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
        # Fetch session to get media_path
        conn = get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM sessions WHERE id = %s", (session_id,))
                session = dict(cur.fetchone())
        finally:
            put_conn(conn)

        media_file = Path(session["media_path"])

        # Step 1: Extract audio
        _update_step(session_id, "extracting_audio")
        print("Processing: extracting audio")
        audio_path = extract_audio(session_id, media_file)

        # Step 2: Preprocess audio (vocal isolation + loudnorm)
        _update_step(session_id, "preprocessing")
        debug_dir = (UPLOAD_DIR / session_id / "debug") if SAVE_INTERMEDIATE else None
        print("Processing: preprocessing audio (vocal isolation + loudnorm)")
        audio_path = preprocess_audio(audio_path, TRANSCRIBE_DEVICE.lower(), debug_dir=debug_dir)

        # Step 3: Transcribe
        _update_step(session_id, "transcribing")
        print("Processing: transcribing audio")
        segments = transcribe_audio(audio_path, debug_dir=debug_dir)

        # Step 4: Merge segments
        _update_step(session_id, "merging")
        print("Processing: merging segments")
        chunks = merge_segments(segments) if segments else []
        if debug_dir and chunks:
            debug_dir.mkdir(parents=True, exist_ok=True)
            out_path = debug_dir / "4_chunks.json"
            with out_path.open("w") as f:
                json.dump(chunks, f, indent=2)
            print(f"[debug] saved {out_path}")

        # Step 5: Embed chunk texts for semantic search
        embeddings = None
        if not DISABLE_SEMANTIC_SEARCH and chunks:
            _update_step(session_id, "embedding")
            print("Processing: embedding chunks for semantic search")
            embed_start = time.time()
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = embed_texts(chunk_texts)
            embed_elapsed = time.time() - embed_start
            if embeddings:
                print(f"Embedded {len(embeddings)} chunks in {embed_elapsed:.1f}s")
            else:
                print("Warning: embedding failed, chunks will have NULL embeddings")

        # Build chunk rows
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

        # Step 6: LLM analysis (fault-tolerant, after chunks committed)
        if not DISABLE_LLM_ANALYSIS and chunk_rows:
            _update_step(session_id, "analyzing")
            print("Processing: running LLM analysis")
            llm_start = time.time()
            llm_chunks = [
                {"id": row[0], "start_ms": row[2], "end_ms": row[3],
                 "text": row[4], "speaker": row[5]}
                for row in chunk_rows
            ]
            duration_ms = max(row[3] for row in chunk_rows)
            try:
                analysis = analyze_session(llm_chunks, duration_ms)
                llm_elapsed = time.time() - llm_start
                if analysis:
                    print(f"LLM analysis complete in {llm_elapsed:.1f}s: "
                          f"{len(analysis['tags'])} tags")
                    conn = get_conn()
                    try:
                        with conn.cursor() as cur:
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
                    "UPDATE sessions SET status = 'ready', audio_path = %s, "
                    "processing_duration_seconds = %s, duration_ms = %s, "
                    "processing_step = NULL WHERE id = %s",
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
                    "UPDATE sessions SET status = 'failed', processing_step = NULL WHERE id = %s",
                    (session_id,),
                )
            conn.commit()
        finally:
            put_conn(conn)
        print(f"Processing failed for session {session_id}: {exc}")


def _poll_loop() -> None:
    """Background loop: pick the oldest queued session and process it."""
    while True:
        try:
            conn = get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM sessions WHERE status = 'queued' "
                        "ORDER BY created_at ASC LIMIT 1"
                    )
                    row = cur.fetchone()
            finally:
                put_conn(conn)

            if row:
                _process_session(row[0])
        except Exception as exc:
            print(f"Worker poll error: {exc}")

        time.sleep(POLL_INTERVAL)


def start_worker() -> None:
    """Launch the polling worker as a daemon thread."""
    t = threading.Thread(target=_poll_loop, daemon=True)
    t.start()
    print("Worker thread started (polling every 2s)")
