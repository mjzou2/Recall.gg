import psycopg2
import psycopg2.extras
import psycopg2.pool
from typing import Dict, List

from fastapi import HTTPException

from app.config import DATA_DIR, PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DBNAME


# Module-level connection pool (initialized in init_storage)
_pool: psycopg2.pool.SimpleConnectionPool = None


def get_pool() -> psycopg2.pool.SimpleConnectionPool:
    """Return the module-level connection pool, creating it if needed."""
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            dbname=PG_DBNAME,
        )
    return _pool


def get_conn():
    """Get a connection from the pool."""
    return get_pool().getconn()


def put_conn(conn):
    """Return a connection to the pool."""
    get_pool().putconn(conn)


def init_storage() -> None:
    """Create folders and tables needed for persistence."""
    for path in (DATA_DIR, DATA_DIR / "uploads", DATA_DIR / "audio"):
        path.mkdir(parents=True, exist_ok=True)

    conn = get_conn()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            # Create sessions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    status TEXT DEFAULT 'created',
                    youtube_url TEXT,
                    media_path TEXT,
                    audio_path TEXT,
                    created_at TEXT,
                    processing_duration_seconds INTEGER,
                    notes TEXT
                )
            """)

            # Safe column additions (Postgres: check information_schema)
            for col, col_type in [
                ("processing_duration_seconds", "INTEGER"),
                ("notes", "TEXT"),
            ]:
                cur.execute("""
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'sessions' AND column_name = %s
                """, (col,))
                if not cur.fetchone():
                    cur.execute(f"ALTER TABLE sessions ADD COLUMN {col} {col_type}")

            # Create chunks table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    session_id TEXT REFERENCES sessions(id),
                    start_ms INTEGER,
                    end_ms INTEGER,
                    text TEXT,
                    notes TEXT,
                    is_bookmarked INTEGER DEFAULT 0,
                    tsv tsvector GENERATED ALWAYS AS (
                        to_tsvector('english', coalesce(text, '') || ' ' || coalesce(notes, ''))
                    ) STORED
                )
            """)

            # Safe column additions for chunks
            for col, col_type, col_default in [
                ("notes", "TEXT", None),
                ("is_bookmarked", "INTEGER", "0"),
            ]:
                cur.execute("""
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'chunks' AND column_name = %s
                """, (col,))
                if not cur.fetchone():
                    default_clause = f" DEFAULT {col_default}" if col_default else ""
                    cur.execute(f"ALTER TABLE chunks ADD COLUMN {col} {col_type}{default_clause}")

            # Check if tsv column exists; add if missing (for existing databases)
            cur.execute("""
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'chunks' AND column_name = 'tsv'
            """)
            if not cur.fetchone():
                cur.execute("""
                    ALTER TABLE chunks ADD COLUMN tsv tsvector
                    GENERATED ALWAYS AS (
                        to_tsvector('english', coalesce(text, '') || ' ' || coalesce(notes, ''))
                    ) STORED
                """)

            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN(tsv)
            """)
    finally:
        put_conn(conn)


def fetch_session(session_id: str) -> Dict:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM sessions WHERE id = %s", (session_id,))
            row = cur.fetchone()
    finally:
        put_conn(conn)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return dict(row)


def fetch_chunks(session_id: str) -> List[Dict]:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, session_id, start_ms, end_ms, text, notes, is_bookmarked FROM chunks WHERE session_id = %s ORDER BY start_ms",
                (session_id,),
            )
            rows = cur.fetchall()
    finally:
        put_conn(conn)
    return [dict(row) for row in rows]


def fetch_chunk(chunk_id: str) -> Dict:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, session_id, start_ms, end_ms, text, notes, is_bookmarked FROM chunks WHERE id = %s",
                (chunk_id,),
            )
            row = cur.fetchone()
    finally:
        put_conn(conn)
    if not row:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return dict(row)
