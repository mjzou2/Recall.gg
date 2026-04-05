import datetime
import shutil
import uuid
from typing import Dict, List

import psycopg2.extras
from fastapi import APIRouter

from app.config import UPLOAD_DIR, AUDIO_DIR
from app.database import get_conn, put_conn, fetch_session, fetch_chunks
from app.services.worker import STEP_ORDER, TOTAL_STEPS
from app.models import (
    SessionCreateRequest,
    SessionUpdateRequest,
    SessionResponse,
)


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse)
def create_session(payload: SessionCreateRequest) -> SessionResponse:
    session_id = str(uuid.uuid4())
    created_at = datetime.datetime.utcnow().isoformat() + "Z"
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions(id, title, youtube_url, notes, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, payload.title, payload.youtube_url, payload.notes, created_at),
            )
        conn.commit()
    finally:
        put_conn(conn)
    return SessionResponse(
        id=session_id,
        title=payload.title,
        status="created",
        youtube_url=payload.youtube_url,
        media_path=None,
        audio_path=None,
        created_at=created_at,
        processing_duration_seconds=None,
        notes=payload.notes,
    )


@router.get("", response_model=List[SessionResponse])
def list_sessions() -> List[SessionResponse]:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC"
            )
            rows = cur.fetchall()
    finally:
        put_conn(conn)
    return [SessionResponse(**dict(row)) for row in rows]


@router.get("/{session_id}/status")
def get_session_status(session_id: str) -> Dict:
    session = fetch_session(session_id)
    step = session.get("processing_step")
    step_number = STEP_ORDER.get(step) if step else None
    total_steps = TOTAL_STEPS if step_number is not None else None
    return {
        "status": session.get("status"),
        "processing_step": step,
        "step_number": step_number,
        "total_steps": total_steps,
    }


@router.get("/{session_id}")
def get_session(session_id: str) -> Dict:
    session = fetch_session(session_id)
    chunks = fetch_chunks(session_id)
    return {"session": session, "chunks": chunks}

@router.patch("/{session_id}", response_model=SessionResponse)
def update_session(session_id: str, payload: SessionUpdateRequest) -> SessionResponse:
    fetch_session(session_id)
    updates = payload.dict(exclude_unset=True)
    if not updates:
        return SessionResponse(**fetch_session(session_id))

    fields = []
    values = []
    for key in ("title", "youtube_url", "notes", "speaker_names"):
        if key in updates:
            fields.append(f"{key} = %s")
            if key == "speaker_names" and updates[key] is not None:
                values.append(psycopg2.extras.Json(updates[key]))
            else:
                values.append(updates[key])

    values.append(session_id)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE sessions SET {', '.join(fields)} WHERE id = %s",
                values,
            )
        conn.commit()
    finally:
        put_conn(conn)

    return SessionResponse(**fetch_session(session_id))

@router.delete("/{session_id}")
def delete_session(session_id: str) -> Dict[str, bool]:
    fetch_session(session_id)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE session_id = %s", (session_id,))
            cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
        conn.commit()
    finally:
        put_conn(conn)

    upload_path = UPLOAD_DIR / session_id
    if upload_path.exists():
        shutil.rmtree(upload_path, ignore_errors=True)

    audio_path = AUDIO_DIR / f"{session_id}.wav"
    if audio_path.exists():
        audio_path.unlink()

    return {"ok": True}
