import os
from pathlib import Path

# Directory paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "app.sqlite"
UPLOAD_DIR = DATA_DIR / "uploads"
AUDIO_DIR = DATA_DIR / "audio"

# Whisper model configuration
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")
TRANSCRIBE_DEVICE = os.environ.get("TRANSCRIBE_DEVICE", "cuda")

# Feature flags
DISABLE_LOL_NORMALIZE = os.environ.get("DISABLE_LOL_NORMALIZE", "0") == "1"
DISABLE_FUZZY_CORRECT = os.environ.get("DISABLE_FUZZY_CORRECT", "0") == "1"
DISABLE_AUDIO_DENOISE = os.environ.get("DISABLE_AUDIO_DENOISE", "0") == "1"

# Fuzzy matching configuration
FUZZY_CATEGORIES = {"champions", "items", "objectives"}
FUZZY_STOPWORDS = {"theres"}  # there's → "theres" matches Thresh at 83
MIN_FUZZY_WORD_LEN = 4
FUZZY_SCORE_CUTOFF = 82
