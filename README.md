# RECALL.GG (local-first MVP)

Scrim/VOD comms transcription + timestamped chunking + search (local-first MVP).

## Dependencies

### System (required)
- ffmpeg (required for video → audio extraction)
- sqlite3 (optional, for inspecting the DB)

Install:
    sudo apt update
    sudo apt install -y ffmpeg sqlite3

### Python (required)
This project uses faster-whisper for transcription (installed in your virtualenv).

Note: if you use GPU transcription, you also need system CUDA/cuDNN (see below).

### Optional: GPU transcription (CUDA)
If you want transcription to run on your NVIDIA GPU:

- WSL2: NVIDIA driver on Windows with WSL GPU support
- Inside Ubuntu (WSL): CUDA Toolkit 12.x (example: 12.9) + cuDNN for CUDA 12 (example: cuDNN 9)

Common packages (after you’ve configured the NVIDIA CUDA apt repo on your system):
    sudo apt update
    sudo apt install -y cuda-toolkit-12-9 cudnn9-cuda-12

## Configuration (env vars)

Backend transcription settings:

- TRANSCRIBE_DEVICE
  - cuda (default, if your machine supports it)
  - cpu (fallback)
- WHISPER_MODEL
  - default: base.en
  - examples: small.en, medium.en

### .env.example (recommended)
Create a file named .env.example at the repo root:

    TRANSCRIBE_DEVICE=cuda
    WHISPER_MODEL=base.en

(Your real .env should be gitignored.)

## Quickstart (Local Dev)

Backend:
    cd ~/projects/vodcomms
    source .venv/bin/activate
    cd backend
    python -m uvicorn main:app --reload --port 8000

Backend API docs:
- http://localhost:8000/docs

Frontend:
    cd ~/projects/vodcomms/frontend
    npm install
    npm run dev

Frontend dev server:
- http://localhost:5173

## Run backend

From repo root:
    cd ~/projects/vodcomms
    source .venv/bin/activate
    cd backend
    python -m uvicorn main:app --reload --port 8000

Force CPU mode:
    cd ~/projects/vodcomms
    source .venv/bin/activate
    cd backend
    TRANSCRIBE_DEVICE=cpu python -m uvicorn main:app --reload --port 8000

Backend API docs:
- http://localhost:8000/docs


## Docker quickstart

From repo root:
    docker compose up --build

Backend API docs:
- http://localhost:8000/docs

Data persists under:
- ./backend/data

## Common issues
- ffmpeg missing: install `ffmpeg` (see Dependencies section).
- backend port already in use: stop the other process or change `--port 8000`.
- SQLite DB location: `backend/data/app.sqlite`.

## Run frontend

    cd ~/projects/vodcomms/frontend
    npm install
    npm run dev

Frontend dev server:
- http://localhost:5173

## Notes
- The app stores transcripts/chunks locally in: backend/data/app.sqlite
- Uploaded media and extracted audio are stored under: backend/data/
