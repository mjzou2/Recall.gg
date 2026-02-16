import datetime
import shutil
import sqlite3
import uuid
from typing import Dict, List

from fastapi import APIRouter

from app.config import DB_PATH, UPLOAD_DIR, AUDIO_DIR
from app.database import fetch_session, fetch_chunks
from app.models import (
    SessionCreateRequest,
    SessionUpdateRequest,
    SessionResponse,
)
from app.utils import row_to_dict


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse)
def create_session(payload: SessionCreateRequest) -> SessionResponse:
    session_id = str(uuid.uuid4())
    created_at = datetime.datetime.utcnow().isoformat() + "Z"
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO sessions(id, title, youtube_url, notes, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, payload.title, payload.youtube_url, payload.notes, created_at),
        )
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
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY datetime(created_at) DESC"
        ).fetchall()
    return [SessionResponse(**row_to_dict(row)) for row in rows]


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
    for key in ("title", "youtube_url", "notes"):
        if key in updates:
            fields.append(f"{key} = ?")
            values.append(updates[key])

    values.append(session_id)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f"UPDATE sessions SET {', '.join(fields)} WHERE id = ?",
            values,
        )

    return SessionResponse(**fetch_session(session_id))

@router.delete("/{session_id}")
def delete_session(session_id: str) -> Dict[str, bool]:
    fetch_session(session_id)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    upload_path = UPLOAD_DIR / session_id
    if upload_path.exists():
        shutil.rmtree(upload_path, ignore_errors=True)

    audio_path = AUDIO_DIR / f"{session_id}.wav"
    if audio_path.exists():
        audio_path.unlink()

    return {"ok": True}
