"""Backfill embeddings for existing chunks that have NULL embedding values.

Usage:
    cd backend
    source .venv/bin/activate
    python -m scripts.backfill_embeddings
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import get_conn, put_conn
from app.services.embedding import embed_texts


BATCH_SIZE = 64


def backfill():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")
            total = cur.fetchone()[0]
    finally:
        put_conn(conn)

    if total == 0:
        print("All chunks already have embeddings. Nothing to do.")
        return

    print(f"Found {total} chunks with NULL embeddings. Starting backfill...")

    processed = 0
    while processed < total:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, text FROM chunks WHERE embedding IS NULL ORDER BY start_ms LIMIT %s",
                    (BATCH_SIZE,),
                )
                rows = cur.fetchall()

            if not rows:
                break

            chunk_ids = [row[0] for row in rows]
            chunk_texts = [row[1] for row in rows]

            embeddings = embed_texts(chunk_texts)
            if embeddings is None:
                print("ERROR: Embedding model failed to load. Aborting backfill.")
                return

            with conn.cursor() as cur:
                for chunk_id, embedding in zip(chunk_ids, embeddings):
                    cur.execute(
                        "UPDATE chunks SET embedding = %s::vector WHERE id = %s",
                        (str(embedding), chunk_id),
                    )
            conn.commit()

            processed += len(rows)
            print(f"  Backfilled {processed}/{total} chunks")
        finally:
            put_conn(conn)

    print(f"Backfill complete. Embedded {processed} chunks.")


if __name__ == "__main__":
    backfill()
