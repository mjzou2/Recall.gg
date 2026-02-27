import uuid

from app.database import get_conn, put_conn


def _insert_chunk(session_id, text, start_ms=0, end_ms=5000, is_bookmarked=0, notes=None):
    """Insert a chunk directly into the database."""
    chunk_id = str(uuid.uuid4())
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO chunks (id, session_id, start_ms, end_ms, text, notes, is_bookmarked)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (chunk_id, session_id, start_ms, end_ms, text, notes, is_bookmarked),
            )
        conn.commit()
    finally:
        put_conn(conn)
    return chunk_id


def test_search_no_results(client):
    resp = client.post("/sessions", json={"title": "Test"})
    session_id = resp.json()["id"]

    resp = client.post(
        f"/sessions/{session_id}/search",
        json={"query": "nonexistent"},
    )
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_search_keyword(client):
    resp = client.post("/sessions", json={"title": "Test"})
    session_id = resp.json()["id"]

    _insert_chunk(session_id, "Baron is spawning soon", start_ms=0, end_ms=5000)
    _insert_chunk(session_id, "Let us recall and buy items", start_ms=5000, end_ms=10000)

    resp = client.post(
        f"/sessions/{session_id}/search",
        json={"query": "Baron"},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 1
    assert "Baron" in results[0]["text"]


def test_search_time_range(client):
    resp = client.post("/sessions", json={"title": "Test"})
    session_id = resp.json()["id"]

    _insert_chunk(session_id, "Early game action", start_ms=0, end_ms=5000)
    _insert_chunk(session_id, "Mid game fight", start_ms=60000, end_ms=65000)
    _insert_chunk(session_id, "Late game baron", start_ms=120000, end_ms=125000)

    resp = client.post(
        f"/sessions/{session_id}/search",
        json={"start_time_ms": 50000, "end_time_ms": 70000},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 1
    assert results[0]["text"] == "Mid game fight"


def test_search_bookmark_filter(client):
    resp = client.post("/sessions", json={"title": "Test"})
    session_id = resp.json()["id"]

    _insert_chunk(session_id, "Normal chunk", start_ms=0, end_ms=5000, is_bookmarked=0)
    _insert_chunk(session_id, "Important chunk", start_ms=5000, end_ms=10000, is_bookmarked=1)

    resp = client.post(
        f"/sessions/{session_id}/search",
        json={"is_bookmarked": True},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 1
    assert results[0]["text"] == "Important chunk"
