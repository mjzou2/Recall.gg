import re
import sqlite3
from typing import Dict, List

from fastapi import APIRouter, HTTPException

from app.config import DB_PATH
from app.database import fetch_session, fetch_chunks, fetch_chunk
from app.models import ChunkResponse, ChunkUpdateRequest, SearchRequest


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
