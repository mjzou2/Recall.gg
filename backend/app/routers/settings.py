from typing import Optional

import app.config as config
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/settings", tags=["settings"])

ALLOWED_WHISPER_MODELS = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v2", "turbo", "large-v3",
]


class SettingsResponse(BaseModel):
    whisper_model: str


class SettingsUpdate(BaseModel):
    whisper_model: Optional[str] = None


@router.get("")
def get_settings() -> SettingsResponse:
    return SettingsResponse(whisper_model=config.WHISPER_MODEL)


@router.put("")
def update_settings(body: SettingsUpdate) -> SettingsResponse:
    if body.whisper_model is not None:
        if body.whisper_model not in ALLOWED_WHISPER_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Allowed: {', '.join(ALLOWED_WHISPER_MODELS)}",
            )
        config.WHISPER_MODEL = body.whisper_model

    return SettingsResponse(whisper_model=config.WHISPER_MODEL)
