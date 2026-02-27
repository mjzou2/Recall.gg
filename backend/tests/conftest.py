import os

# Set test env vars BEFORE importing app modules
# (app/config.py reads os.environ at module import time)
os.environ.setdefault("PG_HOST", "localhost")
os.environ.setdefault("PG_PORT", "5432")
os.environ.setdefault("PG_USER", "recall")
os.environ.setdefault("PG_PASSWORD", "recall")
os.environ.setdefault("PG_DBNAME", "recall")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.database import get_conn, init_storage, put_conn  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def init_db():
    """Initialize database schema once for the entire test session."""
    init_storage()


@pytest.fixture(autouse=True)
def clean_tables():
    """Truncate all data after each test for isolation."""
    yield
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE chunks, sessions CASCADE")
        conn.commit()
    finally:
        put_conn(conn)


@pytest.fixture
def client():
    """FastAPI test client."""
    with TestClient(app) as c:
        yield c
