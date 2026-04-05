from typing import Optional
from pydantic import BaseModel


class SessionCreateRequest(BaseModel):
    title: Optional[str] = None
    youtube_url: Optional[str] = None
    notes: Optional[str] = None


class SessionUpdateRequest(BaseModel):
    title: Optional[str] = None
    youtube_url: Optional[str] = None
    notes: Optional[str] = None
    speaker_names: Optional[dict] = None


class SessionResponse(BaseModel):
    id: str
    title: Optional[str] = None
    status: Optional[str] = None
    youtube_url: Optional[str] = None
    media_path: Optional[str] = None
    audio_path: Optional[str] = None
    created_at: str
    processing_duration_seconds: Optional[int] = None
    duration_ms: Optional[int] = None
    notes: Optional[str] = None
    speaker_names: Optional[dict] = None
    processing_step: Optional[str] = None

class ChunkResponse(BaseModel):
    id: str
    session_id: str
    start_ms: int
    end_ms: int
    text: str
    notes: Optional[str] = None
    is_bookmarked: Optional[int] = 0
    speaker: Optional[str] = None
    youtube_url: Optional[str] = None

class ChunkUpdateRequest(BaseModel):
    notes: Optional[str] = None
    text: Optional[str] = None
    is_bookmarked: Optional[int] = None

class SearchRequest(BaseModel):
    query: str = ""
    limit: int = 20
    start_time_ms: Optional[int] = None
    end_time_ms: Optional[int] = None
    is_bookmarked: Optional[bool] = None
    session_id: Optional[str] = None
