import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Directory paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
AUDIO_DIR = DATA_DIR / "audio"

# Postgres connection params
PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_USER = os.environ.get("PG_USER", "recall")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "recall")
PG_DBNAME = os.environ.get("PG_DBNAME", "recall")

# Whisper model configuration
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")
TRANSCRIBE_DEVICE = os.environ.get("TRANSCRIBE_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "")  # auto, float16, int8, etc. Empty = auto

# Feature flags
DISABLE_LOL_NORMALIZE = os.environ.get("DISABLE_LOL_NORMALIZE", "0") == "1"
DISABLE_FUZZY_CORRECT = os.environ.get("DISABLE_FUZZY_CORRECT", "0") == "1"
DISABLE_AUDIO_DENOISE = os.environ.get("DISABLE_AUDIO_DENOISE", "0") == "1"
DISABLE_SEMANTIC_SEARCH = os.environ.get("DISABLE_SEMANTIC_SEARCH", "0") == "1"
SEMANTIC_SEARCH_THRESHOLD = float(os.environ.get("SEMANTIC_SEARCH_THRESHOLD", "0.5"))
DISABLE_LLM_ANALYSIS = os.environ.get("DISABLE_LLM_ANALYSIS", "0") == "1"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# WhisperX configuration
HF_TOKEN = os.environ.get("HF_TOKEN", "")
_num_speakers = os.environ.get("NUM_SPEAKERS", "")
NUM_SPEAKERS: Optional[int] = int(_num_speakers) if _num_speakers else None
WHISPERX_BATCH_SIZE = int(os.environ.get("WHISPERX_BATCH_SIZE", "16"))
MIN_CHUNK_MS = int(os.environ.get("MIN_CHUNK_MS", "2000"))

# Diarization tuning (all optional; empty = use model defaults)
_diarize_min = os.environ.get("DIARIZE_MIN_SPEAKERS", "")
DIARIZE_MIN_SPEAKERS: Optional[int] = int(_diarize_min) if _diarize_min else None
_diarize_max = os.environ.get("DIARIZE_MAX_SPEAKERS", "")
DIARIZE_MAX_SPEAKERS: Optional[int] = int(_diarize_max) if _diarize_max else None
_diarize_seg = os.environ.get("DIARIZE_SEGMENTATION_THRESHOLD", "")
DIARIZE_SEGMENTATION_THRESHOLD: Optional[float] = float(_diarize_seg) if _diarize_seg else None
_diarize_clust = os.environ.get("DIARIZE_CLUSTERING_THRESHOLD", "")
DIARIZE_CLUSTERING_THRESHOLD: Optional[float] = float(_diarize_clust) if _diarize_clust else None
_diarize_off = os.environ.get("DIARIZE_MIN_DURATION_OFF", "")
DIARIZE_MIN_DURATION_OFF: Optional[float] = float(_diarize_off) if _diarize_off else None
_diarize_on = os.environ.get("DIARIZE_MIN_DURATION_ON", "")
DIARIZE_MIN_DURATION_ON: Optional[float] = float(_diarize_on) if _diarize_on else None
WORD_PAUSE_THRESHOLD_MS = int(os.environ.get("WORD_PAUSE_THRESHOLD", "1500"))

# Debug / development flags
SAVE_INTERMEDIATE = os.environ.get("SAVE_INTERMEDIATE", "0") == "1"

# Fuzzy matching configuration
FUZZY_CATEGORIES = {"champions", "items", "objectives"}
FUZZY_STOPWORDS = {"theres"}  # there's → "theres" matches Thresh at 83
MIN_FUZZY_WORD_LEN = 4
FUZZY_SCORE_CUTOFF = 82
