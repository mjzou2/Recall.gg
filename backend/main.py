import datetime
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from faster_whisper import WhisperModel

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
                status TEXT DEFAULT 'created',
                youtube_url TEXT,
                media_path TEXT,
                audio_path TEXT,
                created_at TEXT
            )
        """
        )
        # Add processing_duration_seconds column if it doesn't exist (for existing databases)
        try:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN processing_duration_seconds INTEGER"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists

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
        conn.execute(
            """
            DROP TRIGGER IF EXISTS chunks_ai
        """
        )
        conn.execute(
            """
            DROP TRIGGER IF EXISTS chunks_ad
        """
        )
        conn.execute(
            """
            DROP TRIGGER IF EXISTS chunks_au
        """
        )
        conn.execute(
            """
            DROP TABLE IF EXISTS chunks_fts
        """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                session_id UNINDEXED,
                content='chunks'
            )
        """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text, session_id)
                VALUES (new.rowid, new.text, new.session_id);
            END;
        """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text, session_id)
                VALUES ('delete', old.rowid, old.text, old.session_id);
            END;
        """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text, session_id)
                VALUES ('delete', old.rowid, old.text, old.session_id);
                INSERT INTO chunks_fts(rowid, text, session_id)
                VALUES (new.rowid, new.text, new.session_id);
            END;
        """
        )
        conn.execute(
            """
            INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')
        """
        )


def row_to_dict(row: sqlite3.Row) -> Dict:
    return {k: row[k] for k in row.keys()}


class SessionCreateRequest(BaseModel):
    title: Optional[str] = None
    youtube_url: Optional[str] = None


class SessionUpdateRequest(BaseModel):
    title: Optional[str] = None
    youtube_url: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    title: Optional[str] = None
    status: Optional[str] = None
    youtube_url: Optional[str] = None
    media_path: Optional[str] = None
    audio_path: Optional[str] = None
    created_at: str
    processing_duration_seconds: Optional[int] = None

class ChunkResponse(BaseModel):
    id: str
    session_id: str
    start_ms: int
    end_ms: int
    text: str

class SearchRequest(BaseModel):
    query: str = ""
    limit: int = 20
    start_time_ms: Optional[int] = None
    end_time_ms: Optional[int] = None

app = FastAPI(title="RECALL.GG", description="Esports comms search MVP")

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
        status="created",
        youtube_url=payload.youtube_url,
        media_path=None,
        audio_path=None,
        created_at=created_at,
        processing_duration_seconds=None,
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

@app.patch("/sessions/{session_id}", response_model=SessionResponse)
def update_session(session_id: str, payload: SessionUpdateRequest) -> SessionResponse:
    fetch_session(session_id)
    updates = payload.dict(exclude_unset=True)
    if not updates:
        return SessionResponse(**fetch_session(session_id))

    fields = []
    values = []
    for key in ("title", "youtube_url"):
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

@app.delete("/sessions/{session_id}")
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

@app.get("/sessions/{session_id}/chunks", response_model=List[ChunkResponse])
def get_chunks(session_id: str) -> List[ChunkResponse]:
    fetch_session(session_id)
    return [ChunkResponse(**c) for c in fetch_chunks(session_id)]

@app.post("/sessions/{session_id}/search")
def search_chunks(session_id: str, payload: SearchRequest) -> Dict[str, List[Dict]]:
    fetch_session(session_id)

    # Extract and validate parameters
    query = payload.query.strip()
    start_time_ms = payload.start_time_ms
    end_time_ms = payload.end_time_ms
    limit = max(0, min(payload.limit, 50))

    # Validate time range
    if start_time_ms is not None and start_time_ms < 0:
        raise HTTPException(status_code=400, detail="Start time cannot be negative")
    if end_time_ms is not None and end_time_ms < 0:
        raise HTTPException(status_code=400, detail="End time cannot be negative")
    if start_time_ms is not None and end_time_ms is not None and start_time_ms >= end_time_ms:
        raise HTTPException(status_code=400, detail="End time must be after start time")

    # Handle one-sided ranges
    if start_time_ms is not None and end_time_ms is None:
        end_time_ms = 999999999999  # Very large value (over 11 days)
    if end_time_ms is not None and start_time_ms is None:
        start_time_ms = 0

    # Return empty if no filters provided
    if not query and start_time_ms is None and end_time_ms is None:
        return {"results": []}

    if limit == 0:
        return {"results": []}

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Case 1: Keyword only (existing behavior)
        if query and start_time_ms is None:
            tokens = [token for token in query.split() if token]
            fts_query = " ".join(f"{token}*" for token in tokens)

            rows = conn.execute(
                """
                SELECT c.id, c.start_ms, c.end_ms, c.text
                FROM chunks_fts f
                JOIN chunks c ON c.rowid = f.rowid
                WHERE f.session_id = ?
                  AND chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
                """,
                (session_id, fts_query, limit),
            ).fetchall()

        # Case 2: Time range only
        elif not query and start_time_ms is not None:
            rows = conn.execute(
                """
                SELECT id, start_ms, end_ms, text
                FROM chunks
                WHERE session_id = ?
                  AND start_ms < ?
                  AND end_ms > ?
                ORDER BY start_ms
                LIMIT ?
                """,
                (session_id, end_time_ms, start_time_ms, limit),
            ).fetchall()

        # Case 3: Combined keyword + time range
        else:  # query and start_time_ms is not None
            tokens = [token for token in query.split() if token]
            fts_query = " ".join(f"{token}*" for token in tokens)

            rows = conn.execute(
                """
                SELECT c.id, c.start_ms, c.end_ms, c.text
                FROM chunks_fts f
                JOIN chunks c ON c.rowid = f.rowid
                WHERE f.session_id = ?
                  AND chunks_fts MATCH ?
                  AND c.start_ms < ?
                  AND c.end_ms > ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
                """,
                (session_id, fts_query, end_time_ms, start_time_ms, limit),
            ).fetchall()

    results = [
        {
            "id": row["id"],
            "start_ms": row["start_ms"],
            "end_ms": row["end_ms"],
            "text": row["text"],
        }
        for row in rows
    ]

    return {"results": results}

@app.post("/sessions/{session_id}/media")
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


_whisper_model: Optional[WhisperModel] = None


def get_transcriber() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        device = os.environ.get("TRANSCRIBE_DEVICE", "cuda").lower()
        is_cpu = device == "cpu"
        compute_type = "int8" if is_cpu else "float16"
        model_name = os.environ.get("WHISPER_MODEL", "base.en")

        _whisper_model = WhisperModel(
            model_name,
            device="cpu" if is_cpu else "cuda",
            compute_type=compute_type,
        )
    return _whisper_model


def normalize_lol_text(text: str) -> str:
    replacements = [
        (r"\bharold\b", "herald"),
        (r"\bword(s)?\b", r"ward\1"),
        (r"\bdrakes?\b", "drake"),
        (r"\bksante\b", "K'Sante"),
        (r"\bk sante\b", "K'Sante"),
        (r"\btp\b", "TP"),
    ]
    normalized = text
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized

def load_term_bank() -> Dict[str, List[str]]:
    """Load the League of Legends term bank from JSON."""
    term_bank_path = Path(__file__).resolve().parent / "term_bank.json"
    try:
        with open(term_bank_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Term bank not found at {term_bank_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse term bank: {e}")
        return {}

def transcribe_audio(audio_path: Path) -> List[Dict]:
    # Load term bank
    term_bank = load_term_bank()
    
    # Build hotwords list: all objectives + top common_comms terms
    hotwords_list = []
    if term_bank:
        # Add all objectives
        hotwords_list.extend(term_bank.get("objectives", []))
        # Add high-frequency comms terms (pick first 15)
        hotwords_list.extend(term_bank.get("common_comms", [])[:15])
    
    # Fallback if term bank fails to load
    if not hotwords_list:
        hotwords_list = [
            "baron", "herald", "grubs", "dragon", "drake", "ward", "reset", 
            "tp", "teleport", "flash", "smite", "invade", "dive", "prio", "push"
        ]
    
    # Convert list to comma-separated string
    hotwords = ", ".join(hotwords_list)
    
    # Build initial_prompt: natural sentences with key terms
    initial_prompt = ""
    if term_bank:
        objectives = term_bank.get("objectives", [])[:5]  # First 5 objectives
        champions = term_bank.get("champions", [])[:4]    # First 4 champions
        comms = term_bank.get("common_comms", [])[:4]     # First 4 comms terms
        
        initial_prompt = (
            f"The team secured {objectives[0] if objectives else 'Baron'} and warded river. "
            f"{champions[0] if champions else 'The mid laner'} and {champions[1] if len(champions) > 1 else 'the jungler'} "
            f"rotated mid while tracking the enemy. They need to {comms[0] if comms else 'reset'} and get vision."
        )
    
    normalize_enabled = os.environ.get("DISABLE_LOL_NORMALIZE", "0") != "1"
    device = os.environ.get("TRANSCRIBE_DEVICE", "cuda").lower()
    model_name = os.environ.get("WHISPER_MODEL", "base.en")
    
    print(
        "Transcribing with model=%s device=%s hotwords=%s initial_prompt=%s normalize=%s"
        % (model_name, device, bool(hotwords), bool(initial_prompt), normalize_enabled)
    )
    
    try:
        segments, _info = get_transcriber().transcribe(
            str(audio_path),
            vad_filter=True,
            hotwords=hotwords,
            initial_prompt=initial_prompt, 
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Transcription failed: {exc}"
        ) from exc

    results: List[Dict] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if normalize_enabled:
            text = normalize_lol_text(text)
        results.append(
            {
                "start_ms": int(segment.start * 1000),
                "end_ms": int(segment.end * 1000),
                "text": text,
            }
        )
    return results


def merge_segments(segments: List[Dict], max_chunk_ms: int = 35000) -> List[Dict]:
    merged: List[Dict] = []
    current: Optional[Dict] = None

    for segment in segments:
        if current is None:
            current = segment.copy()
            continue

        if segment["end_ms"] - current["start_ms"] <= max_chunk_ms:
            current["end_ms"] = segment["end_ms"]
            current["text"] = f"{current['text']} {segment['text']}"
        else:
            merged.append(current)
            current = segment.copy()

    if current is not None:
        merged.append(current)

    return merged


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
