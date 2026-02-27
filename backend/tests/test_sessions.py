def test_create_session(client):
    resp = client.post("/sessions", json={"title": "Test Scrim"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"]
    assert data["title"] == "Test Scrim"
    assert data["status"] == "created"
    assert data["created_at"]


def test_list_sessions(client):
    client.post("/sessions", json={"title": "Session A"})
    client.post("/sessions", json={"title": "Session B"})
    resp = client.get("/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    # Newest first
    assert data[0]["title"] == "Session B"
    assert data[1]["title"] == "Session A"


def test_get_session(client):
    create_resp = client.post("/sessions", json={"title": "Get Test"})
    session_id = create_resp.json()["id"]

    resp = client.get(f"/sessions/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session"]["id"] == session_id
    assert data["session"]["title"] == "Get Test"
    assert data["chunks"] == []


def test_update_session(client):
    create_resp = client.post("/sessions", json={"title": "Original"})
    session_id = create_resp.json()["id"]

    resp = client.patch(
        f"/sessions/{session_id}",
        json={"title": "Updated", "youtube_url": "https://youtube.com/watch?v=abc"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Updated"
    assert data["youtube_url"] == "https://youtube.com/watch?v=abc"


def test_delete_session(client):
    create_resp = client.post("/sessions", json={"title": "To Delete"})
    session_id = create_resp.json()["id"]

    resp = client.delete(f"/sessions/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    resp = client.get(f"/sessions/{session_id}")
    assert resp.status_code == 404
