import sqlite3
from typing import Dict, List

from fastapi import HTTPException

from app.config import DB_PATH, DATA_DIR
from app.utils import row_to_dict


def init_storage() -> None:
    """Create folders and tables needed for local persistence."""
    for path in (DATA_DIR, DATA_DIR / "uploads", DATA_DIR / "audio"):
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
