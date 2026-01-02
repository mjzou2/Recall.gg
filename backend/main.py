import datetime
import shutil
import sqlite3
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "app.sqlite"
UPLOAD_DIR = DATA_DIR / "uploads"
AUDIO_DIR = DATA_DIR / "audio"


def init_storage() -> None:
    """Create folders and tables needed for local persistence."""
    for path in (DATA_DIR, UPLOAD_DIR, AUDIO_DIR):
        path.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                youtube_url TEXT,
                media_path TEXT,
                audio_path TEXT,
                created_at TEXT
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                start_ms INTEGER,
                end_ms INTEGER,
                text TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id)
        """
        )


def row_to_dict(row: sqlite3.Row) -> Dict:
    return {k: row[k] for k in row.keys()}


class SessionCreateRequest(BaseModel):
    title: Optional[str] = None
    youtube_url: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    title: Optional[str]
    youtube_url: Optional[str]
    media_path: Optional[str]
    audio_path: Optional[str]
    created_at: str


app = FastAPI(title="vodcomms MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_storage()


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


def fetch_session(session_id: str) -> Dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return row_to_dict(row)


def fetch_chunks(session_id: str) -> List[Dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM chunks WHERE session_id = ? ORDER BY start_ms",
            (session_id,),
        ).fetchall()
    return [row_to_dict(row) for row in rows]


@app.post("/sessions", response_model=SessionResponse)
def create_session(payload: SessionCreateRequest) -> SessionResponse:
    session_id = str(uuid.uuid4())
    created_at = datetime.datetime.utcnow().isoformat() + "Z"
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO sessions(id, title, youtube_url, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, payload.title, payload.youtube_url, created_at),
        )
    return SessionResponse(
        id=session_id,
        title=payload.title,
        youtube_url=payload.youtube_url,
        media_path=None,
        audio_path=None,
        created_at=created_at,
    )


@app.get("/sessions", response_model=List[SessionResponse])
def list_sessions() -> List[SessionResponse]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY datetime(created_at) DESC"
        ).fetchall()
    return [SessionResponse(**row_to_dict(row)) for row in rows]


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> Dict:
    session = fetch_session(session_id)
    chunks = fetch_chunks(session_id)
    return {"session": session, "chunks": chunks}


@app.post("/sessions/{session_id}/media")
async def upload_media(session_id: str, file: UploadFile = File(...)) -> Dict:
    session = fetch_session(session_id)
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    dest_dir = UPLOAD_DIR / session_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(file.filename).name

    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE sessions SET media_path = ?, audio_path = NULL WHERE id = ?",
            (str(dest_path), session_id),
        )

    return {
        "session": session_id,
        "stored_path": str(dest_path),
        "original_filename": file.filename,
    }


def extract_audio(session_id: str, media_path: Path) -> Path:
    audio_path = AUDIO_DIR / f"{session_id}.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip extraction if file is already an audio-only type.
    if media_path.suffix.lower() in {".wav", ".mp3", ".m4a", ".ogg", ".flac"}:
        return media_path

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(media_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                str(audio_path),
            ],
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


@app.post("/sessions/{session_id}/process")
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

    audio_path = extract_audio(session_id, media_file)

    dummy_chunks = [
        (
            str(uuid.uuid4()),
            session_id,
            0,
            15000,
            "Intro and setup for the round.",
        ),
        (
            str(uuid.uuid4()),
            session_id,
            15000,
            30000,
            "Key play-by-play comms.",
        ),
        (
            str(uuid.uuid4()),
            session_id,
            30000,
            45000,
            "Post-round recap and callouts.",
        ),
    ]

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
        conn.executemany(
            """
            INSERT INTO chunks(id, session_id, start_ms, end_ms, text)
            VALUES (?, ?, ?, ?, ?)
            """,
            dummy_chunks,
        )
        conn.execute(
            "UPDATE sessions SET audio_path = ? WHERE id = ?",
            (str(audio_path), session_id),
        )

    return {
        "session": session_id,
        "audio_path": str(audio_path),
        "chunks": fetch_chunks(session_id),
    }
