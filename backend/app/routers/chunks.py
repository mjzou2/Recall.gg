import re
from typing import Dict, List

import psycopg2.extras
from fastapi import APIRouter, HTTPException

from app.config import DISABLE_SEMANTIC_SEARCH, SEMANTIC_SEARCH_THRESHOLD
from app.database import get_conn, put_conn, fetch_session, fetch_chunks, fetch_chunk
from app.models import ChunkResponse, ChunkUpdateRequest, SearchRequest
from app.services.embedding import embed_text


router = APIRouter(tags=["chunks"])


@router.get("/sessions/{session_id}/chunks", response_model=List[ChunkResponse])
def get_chunks(session_id: str) -> List[ChunkResponse]:
    fetch_session(session_id)
    return [ChunkResponse(**c) for c in fetch_chunks(session_id)]

@router.post("/sessions/{session_id}/search")
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

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            keyword_results = []
            semantic_results = []

            if query:
                # --- Keyword search (existing behavior, unchanged) ---
                tokens = [token for token in query.split() if token]
                sanitized_tokens = [re.sub(r"['\"\(\)]", "", token) for token in tokens if token]
                tsquery_parts = [f"{token}:*" for token in sanitized_tokens if token]
                tsquery_str = " & ".join(tsquery_parts)

                where_clauses = ["session_id = %s", "tsv @@ to_tsquery('english', %s)"]
                params: list = [session_id, tsquery_str]

                if start_time_ms is not None:
                    where_clauses.append("start_ms < %s")
                    where_clauses.append("end_ms > %s")
                    params.extend([end_time_ms, start_time_ms])

                if is_bookmarked is True:
                    where_clauses.append("is_bookmarked = 1")

                keyword_sql = f"""
                    SELECT id, session_id, start_ms, end_ms, text, notes, is_bookmarked, speaker
                    FROM chunks
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY start_ms
                    LIMIT %s
                """
                params.append(limit)
                cur.execute(keyword_sql, params)
                keyword_results = cur.fetchall()

                # --- Semantic search (additive) ---
                if not DISABLE_SEMANTIC_SEARCH:
                    query_embedding = embed_text(query)
                    if query_embedding is not None:
                        sem_where = ["session_id = %s", "embedding IS NOT NULL",
                                     "embedding <=> %s::vector < %s"]
                        query_emb_str = str(query_embedding)
                        sem_params: list = [session_id, query_emb_str, SEMANTIC_SEARCH_THRESHOLD]

                        if start_time_ms is not None:
                            sem_where.append("start_ms < %s")
                            sem_where.append("end_ms > %s")
                            sem_params.extend([end_time_ms, start_time_ms])

                        if is_bookmarked is True:
                            sem_where.append("is_bookmarked = 1")

                        sem_sql = f"""
                            SELECT id, session_id, start_ms, end_ms, text, notes, is_bookmarked, speaker
                            FROM chunks
                            WHERE {' AND '.join(sem_where)}
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s
                        """
                        sem_params.append(query_emb_str)
                        sem_params.append(20)
                        cur.execute(sem_sql, sem_params)
                        semantic_results = cur.fetchall()

            else:
                # --- No query: time-range / bookmark only (unchanged) ---
                where_clauses = ["session_id = %s"]
                params = [session_id]

                if start_time_ms is not None:
                    where_clauses.append("start_ms < %s")
                    where_clauses.append("end_ms > %s")
                    params.extend([end_time_ms, start_time_ms])

                if is_bookmarked is True:
                    where_clauses.append("is_bookmarked = 1")

                query_sql = f"""
                    SELECT id, session_id, start_ms, end_ms, text, notes, is_bookmarked, speaker
                    FROM chunks
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY start_ms
                    LIMIT %s
                """
                params.append(limit)
                cur.execute(query_sql, params)
                keyword_results = cur.fetchall()
    finally:
        put_conn(conn)

    # Merge and deduplicate (keyword results take priority)
    seen_ids = set()
    merged = []
    for row in keyword_results:
        if row["id"] not in seen_ids:
            seen_ids.add(row["id"])
            merged.append(row)
    for row in semantic_results:
        if row["id"] not in seen_ids:
            seen_ids.add(row["id"])
            merged.append(row)

    # Sort combined results chronologically
    merged.sort(key=lambda r: r["start_ms"])

    results = [
        {
            "id": row["id"],
            "start_ms": row["start_ms"],
            "end_ms": row["end_ms"],
            "text": row["text"],
            "notes": row["notes"],
            "is_bookmarked": row["is_bookmarked"],
            "speaker": row.get("speaker"),
        }
        for row in merged
    ]

    return {"results": results}

@router.patch("/chunks/{chunk_id}", response_model=ChunkResponse)
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
        set_clauses.append("notes = %s")
        values.append(updates["notes"])
    if "text" in updates:
        set_clauses.append("text = %s")
        values.append(updates["text"])
    if "is_bookmarked" in updates:
        set_clauses.append("is_bookmarked = %s")
        values.append(updates["is_bookmarked"])

    # Re-embed if text changed
    if "text" in updates and updates["text"] is not None and not DISABLE_SEMANTIC_SEARCH:
        new_embedding = embed_text(updates["text"])
        if new_embedding is not None:
            set_clauses.append("embedding = %s::vector")
            values.append(str(new_embedding))

    values.append(chunk_id)

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE chunks SET {', '.join(set_clauses)} WHERE id = %s",
                values,
            )
        conn.commit()
    finally:
        put_conn(conn)

    return ChunkResponse(**fetch_chunk(chunk_id))
