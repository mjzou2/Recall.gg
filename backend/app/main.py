from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_storage
from app.routers import sessions, chunks, media, settings


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


# Include routers
app.include_router(sessions.router)
app.include_router(chunks.router)
app.include_router(media.router)
app.include_router(settings.router)
