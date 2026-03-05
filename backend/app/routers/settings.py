import json
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/settings", tags=["settings"])

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "settings.json")

ALLOWED_WHISPER_MODELS = [
    "tiny", "base", "small", "medium", "large-v2", "turbo", "large-v3",
]


class SettingsResponse(BaseModel):
    whisper_model: str


class SettingsUpdate(BaseModel):
    whisper_model: Optional[str] = None


def _read_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return {}


def _write_settings(data: dict) -> None:
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _get_whisper_model() -> str:
    settings = _read_settings()
    return settings.get("whisper_model", os.environ.get("WHISPER_MODEL", "base.en"))


@router.get("")
def get_settings() -> SettingsResponse:
    return SettingsResponse(whisper_model=_get_whisper_model())


@router.put("")
def update_settings(body: SettingsUpdate) -> SettingsResponse:
    settings = _read_settings()

    if body.whisper_model is not None:
        if body.whisper_model not in ALLOWED_WHISPER_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Allowed: {', '.join(ALLOWED_WHISPER_MODELS)}",
            )
        settings["whisper_model"] = body.whisper_model

    _write_settings(settings)
    return SettingsResponse(whisper_model=settings.get("whisper_model", _get_whisper_model()))
