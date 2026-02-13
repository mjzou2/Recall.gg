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
from rapidfuzz import fuzz, process

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

        # Add notes column if it doesn't exist (for existing databases)
        try:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN notes TEXT"
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
        # Add notes column if it doesn't exist (for existing databases)
        try:
            conn.execute(
                "ALTER TABLE chunks ADD COLUMN notes TEXT"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add is_bookmarked column if it doesn't exist (for existing databases)
        try:
            conn.execute(
                "ALTER TABLE chunks ADD COLUMN is_bookmarked INTEGER DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists

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
                notes,
                session_id UNINDEXED,
                content='chunks'
            )
        """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text, notes, session_id)
                VALUES (new.rowid, new.text, new.notes, new.session_id);
            END;
        """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text, notes, session_id)
                VALUES ('delete', old.rowid, old.text, old.notes, old.session_id);
            END;
        """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text, notes, session_id)
                VALUES ('delete', old.rowid, old.text, old.notes, old.session_id);
                INSERT INTO chunks_fts(rowid, text, notes, session_id)
                VALUES (new.rowid, new.text, new.notes, new.session_id);
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
    notes: Optional[str] = None


class SessionUpdateRequest(BaseModel):
    title: Optional[str] = None
    youtube_url: Optional[str] = None
    notes: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    title: Optional[str] = None
    status: Optional[str] = None
    youtube_url: Optional[str] = None
    media_path: Optional[str] = None
    audio_path: Optional[str] = None
    created_at: str
    processing_duration_seconds: Optional[int] = None
    notes: Optional[str] = None

class ChunkResponse(BaseModel):
    id: str
    session_id: str
    start_ms: int
    end_ms: int
    text: str
    notes: Optional[str] = None
    is_bookmarked: Optional[int] = 0

class ChunkUpdateRequest(BaseModel):
    notes: Optional[str] = None
    text: Optional[str] = None
    is_bookmarked: Optional[int] = None

class SearchRequest(BaseModel):
    query: str = ""
    limit: int = 20
    start_time_ms: Optional[int] = None
    end_time_ms: Optional[int] = None
    is_bookmarked: Optional[bool] = None

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


def fetch_chunk(chunk_id: str) -> Dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return row_to_dict(row)


@app.post("/sessions", response_model=SessionResponse)
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
    is_bookmarked = payload.is_bookmarked
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
    if not query and start_time_ms is None and end_time_ms is None and is_bookmarked is None:
        return {"results": []}

    if limit == 0:
        return {"results": []}

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Determine if we need FTS search
        use_fts = bool(query)

        if use_fts:
            # Prepare FTS query
            tokens = [token for token in query.split() if token]
            sanitized_tokens = [re.sub(r"['\"\(\)]", "", token) for token in tokens if token]
            fts_query = " ".join(f"{token}*" for token in sanitized_tokens if token)

            # Build WHERE clauses for additional filters
            where_clauses = ["f.session_id = ?", "chunks_fts MATCH ?"]
            params = [session_id, fts_query]

            if start_time_ms is not None:
                where_clauses.append("c.start_ms < ?")
                where_clauses.append("c.end_ms > ?")
                params.extend([end_time_ms, start_time_ms])

            if is_bookmarked is True:
                where_clauses.append("c.is_bookmarked = 1")

            query_sql = f"""
                SELECT c.id, c.start_ms, c.end_ms, c.text, c.notes, c.is_bookmarked
                FROM chunks_fts f
                JOIN chunks c ON c.rowid = f.rowid
                WHERE {' AND '.join(where_clauses)}
                ORDER BY c.start_ms
                LIMIT ?
            """
            params.append(limit)
            rows = conn.execute(query_sql, params).fetchall()

        else:
            # Direct query on chunks table
            where_clauses = ["session_id = ?"]
            params = [session_id]

            if start_time_ms is not None:
                where_clauses.append("start_ms < ?")
                where_clauses.append("end_ms > ?")
                params.extend([end_time_ms, start_time_ms])

            if is_bookmarked is True:
                where_clauses.append("is_bookmarked = 1")

            query_sql = f"""
                SELECT id, start_ms, end_ms, text, notes, is_bookmarked
                FROM chunks
                WHERE {' AND '.join(where_clauses)}
                ORDER BY start_ms
                LIMIT ?
            """
            params.append(limit)
            rows = conn.execute(query_sql, params).fetchall()

    results = [
        {
            "id": row["id"],
            "start_ms": row["start_ms"],
            "end_ms": row["end_ms"],
            "text": row["text"],
            "notes": row["notes"],
            "is_bookmarked": row["is_bookmarked"],
        }
        for row in rows
    ]

    return {"results": results}

@app.patch("/chunks/{chunk_id}", response_model=ChunkResponse)
def update_chunk(chunk_id: str, payload: ChunkUpdateRequest) -> ChunkResponse:
    """Update a chunk's notes and/or text field."""
    fetch_chunk(chunk_id)  # Verify chunk exists
    updates = payload.dict(exclude_unset=True)
    if not updates:
        return ChunkResponse(**fetch_chunk(chunk_id))

    # Validate and prepare text if provided
    if "text" in updates:
        text = updates["text"]
        if text is not None:
            text = text.strip()
            if not text:
                raise HTTPException(status_code=400, detail="Text cannot be empty")
            if len(text) > 1000:
                raise HTTPException(status_code=400, detail="Text cannot exceed 1000 characters")
            updates["text"] = text

    # Validate notes if provided (existing validation)
    if "notes" in updates and updates["notes"] is not None:
        notes = updates["notes"].strip()
        updates["notes"] = notes

    # Validate is_bookmarked if provided
    if "is_bookmarked" in updates and updates["is_bookmarked"] is not None:
        if updates["is_bookmarked"] not in (0, 1):
            raise HTTPException(status_code=400, detail="is_bookmarked must be 0 or 1")

    # Build dynamic UPDATE query
    set_clauses = []
    values = []
    if "notes" in updates:
        set_clauses.append("notes = ?")
        values.append(updates["notes"])
    if "text" in updates:
        set_clauses.append("text = ?")
        values.append(updates["text"])
    if "is_bookmarked" in updates:
        set_clauses.append("is_bookmarked = ?")
        values.append(updates["is_bookmarked"])

    values.append(chunk_id)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f"UPDATE chunks SET {', '.join(set_clauses)} WHERE id = ?",
            values,
        )

    return ChunkResponse(**fetch_chunk(chunk_id))

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

    # Light denoise filter chain for noisy Discord comms:
    #   highpass=f=200   — cut low rumble/hum below 200Hz
    #   lowpass=f=3500   — cut hiss/static above 3500Hz (voice comms don't need higher)
    #   afftdn=nf=-25    — gentle FFT-based noise reduction (lower = more aggressive)
    denoise = os.environ.get("DISABLE_AUDIO_DENOISE", "0") != "1"
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
        # Common Whisper mishears
        (r"\bharold\b", "herald"),
        (r"\bword(s|ing|ed)?\b", r"ward\1"),
        # Severe mishears fuzzy matching can't catch (score < 82)
        (r"\bcass?ante\b", "K'Sante"),
        (r"\baura\b", "Aurora"),
        (r"\brookong\b", "Wukong"),
        (r"\bni[kc]os?\b", "Neeko"),  # matches: niko, nico, nikos, nicos
        (r"\bflush\b", "flash"),
        (r"\bani\b", "Annie"),
        # Space-separated apostrophe champions (parts too short for fuzzy)
        (r"\bk sante\b", "K'Sante"),
        (r"\bksante\b", "K'Sante"),
        (r"\bkha zix\b", "Kha'Zix"),
        (r"\bcho gath\b", "Cho'Gath"),
        (r"\bbel veth\b", "Bel'Veth"),
        (r"\bvel koz\b", "Vel'Koz"),
        (r"\bkog maw\b", "Kog'Maw"),
        (r"\brek sai\b", "Rek'Sai"),
        (r"\bkai sa\b", "Kai'Sa"),
        # Short word mishears (< 4 chars, fuzzy skips)
        (r"\bchen\b", "Shen"),
        (r"\bjacks?\b", "Jax"),  # matches: jax, jack, jacks, jack's (apostrophe stripped by re.IGNORECASE)
        # Short abbreviations (< 4 chars, fuzzy skips)
        (r"\b[dt]p\b", "TP"),  # matches: tp, dp
        (r"\bcc\b", "CC"),
        (r"\bcs\b", "CS"),
        (r"\badc\b", "ADC"),
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

FUZZY_CATEGORIES = {"champions", "items", "objectives"}

# Common English words whose normalized form (lowercase, no apostrophes) scores >= 82
# against a champion/item/objective name.  These must never be fuzzy-replaced.
# Discovered via transcription_errors.tsv testing.
FUZZY_STOPWORDS = {"theres"}  # there's → "theres" matches Thresh at 83


def _build_fuzzy_lookup(term_bank: Dict[str, List[str]]) -> Dict[int, Dict[str, str]]:
    """Build lookup tables for fuzzy matching, keyed by word count.

    Only includes champions, items, and objectives — common_comms, locations,
    and mechanics are skipped because they contain common English words that
    cause false positives (e.g. "taking" → "tracking").

    Returns dict mapping word_count -> {normalized_form: original_term}.
    e.g. {1: {"ksante": "K'Sante"}, 2: {"lee sin": "Lee Sin"}}
    """
    lookup: Dict[int, Dict[str, str]] = {}
    seen_normalized: set = set()

    for category, terms in term_bank.items():
        if category not in FUZZY_CATEGORIES:
            continue
        for term in terms:
            normalized = term.lower().replace("'", "")
            if normalized in seen_normalized:
                continue
            seen_normalized.add(normalized)
            word_count = len(normalized.split())
            if word_count not in lookup:
                lookup[word_count] = {}
            lookup[word_count][normalized] = term

    return lookup


MIN_FUZZY_WORD_LEN = 4
FUZZY_SCORE_CUTOFF = 82


def fuzzy_correct_text(text: str, fuzzy_lookup: Dict[int, Dict[str, str]]) -> str:
    """Fuzzy-match words against the term bank and replace confident matches.

    Two-pass approach:
      Pass 1: Multi-word n-grams (longest first)
      Pass 2: Single words (length >= MIN_FUZZY_WORD_LEN)
    """
    if not fuzzy_lookup:
        return text

    words = text.split()
    if not words:
        return text

    matched = [False] * len(words)
    output = list(words)

    # --- Pass 1: Multi-word matching (longest n-grams first) ---
    multi_sizes = sorted([n for n in fuzzy_lookup if n > 1], reverse=True)

    for n in multi_sizes:
        terms_for_n = fuzzy_lookup[n]
        if not terms_for_n:
            continue
        choices = list(terms_for_n.keys())

        i = 0
        while i <= len(words) - n:
            if any(matched[i:i + n]):
                i += 1
                continue

            raw = " ".join(words[i:i + n])
            window = raw.strip(".,!?;:\"'()[]").lower().replace("'", "")
            result = process.extractOne(
                window,
                choices,
                scorer=fuzz.ratio,
                score_cutoff=FUZZY_SCORE_CUTOFF,
            )

            if result is not None:
                best_match, score, _index = result
                original_term = terms_for_n[best_match]
                replacement_words = original_term.split()
                # Preserve trailing punctuation from the last word in the window
                last_word = words[i + n - 1]
                trailing_punct = last_word[len(last_word.rstrip(".,!?;:\"'()[]")):]
                for j in range(n):
                    if j < len(replacement_words):
                        output[i + j] = replacement_words[j]
                    else:
                        output[i + j] = ""
                    matched[i + j] = True
                if trailing_punct and replacement_words:
                    last_idx = i + min(n, len(replacement_words)) - 1
                    output[last_idx] += trailing_punct
                i += n
            else:
                i += 1

    # --- Pass 2: Single-word matching ---
    single_terms = fuzzy_lookup.get(1, {})
    if single_terms:
        choices_single = list(single_terms.keys())

        for i, word in enumerate(words):
            if matched[i]:
                continue

            stripped = word.strip(".,!?;:\"'()[]")
            if len(stripped) < MIN_FUZZY_WORD_LEN:
                continue

            normalized = stripped.lower().replace("'", "")
            if normalized in FUZZY_STOPWORDS:
                continue
            result = process.extractOne(
                normalized,
                choices_single,
                scorer=fuzz.ratio,
                score_cutoff=FUZZY_SCORE_CUTOFF,
            )

            if result is not None:
                best_match, score, _index = result
                original_term = single_terms[best_match]
                prefix = word[:len(word) - len(word.lstrip(".,!?;:\"'()[]"))]
                suffix = word[len(word.rstrip(".,!?;:\"'()[]")):]
                output[i] = prefix + original_term + suffix
                matched[i] = True

    return " ".join(w for w in output if w)


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
    fuzzy_enabled = os.environ.get("DISABLE_FUZZY_CORRECT", "0") != "1"
    fuzzy_lookup = _build_fuzzy_lookup(term_bank) if fuzzy_enabled else {}
    device = os.environ.get("TRANSCRIBE_DEVICE", "cuda").lower()
    model_name = os.environ.get("WHISPER_MODEL", "base.en")

    print(
        "Transcribing with model=%s device=%s hotwords=%s initial_prompt=%s normalize=%s fuzzy=%s"
        % (model_name, device, bool(hotwords), bool(initial_prompt), normalize_enabled, fuzzy_enabled)
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
        if fuzzy_enabled and fuzzy_lookup:
            text = fuzzy_correct_text(text, fuzzy_lookup)
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
