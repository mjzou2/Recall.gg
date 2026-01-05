## Dependencies

### System
- ffmpeg (required for video → audio extraction)
- sqlite3 (optional, for inspecting the DB)
- faster-whisper (Python package for transcription; install in your virtualenv)

Install:
```bash
sudo apt update
sudo apt install -y ffmpeg sqlite3

## Run backend

cd .../vodcomms/backend

source .venv/bin/activate

python -m uvicorn main:app --reload --port 8000

## Run frontend

cd .../vodcomms/frontend

npm run dev
