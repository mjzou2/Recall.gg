"""Backfill duration_ms for sessions by computing max(end_ms) from their chunks.

Usage:
    cd backend
    source .venv/bin/activate
    python -m scripts.backfill_duration
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import get_conn, put_conn


def backfill():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT s.id, MAX(c.end_ms) AS max_end_ms
                FROM sessions s
                JOIN chunks c ON c.session_id = s.id
                WHERE s.duration_ms IS NULL
                GROUP BY s.id
            """)
            rows = cur.fetchall()

        if not rows:
            print("All sessions already have duration_ms. Nothing to do.")
            return

        print(f"Found {len(rows)} sessions with NULL duration_ms. Backfilling...")

        with conn.cursor() as cur:
            for session_id, max_end_ms in rows:
                cur.execute(
                    "UPDATE sessions SET duration_ms = %s WHERE id = %s",
                    (max_end_ms, session_id),
                )
                print(f"  {session_id}: {max_end_ms} ms")
        conn.commit()

        print(f"Backfill complete. Updated {len(rows)} sessions.")
    finally:
        put_conn(conn)


if __name__ == "__main__":
    backfill()
