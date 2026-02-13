# RECALL.GG

**A local-first esports VOD comms search and analytics tool for League of Legends**

---

## The Problem

In esports, coaches and teams need to review and discuss key moments in their practice. But finding these moments usually means manually scrubbing through an entire VOD, trying to remember what/when important things happened. Review sessions are often unstructured and may not always complete in the allotted review time.

## The Solution (Now)

1. **Upload** your scrim VOD with audio
2. **Process** with GPU-accelerated transcription in minutes
3. **Search** for any keyword - "Baron", "dive mid", "flash", "reset"
4. **Jump** directly to that moment in the embedded YouTube player with one click

Gone are the days of manually scrubbing through a 5+ hour VOD. Now you can find the moments that matter in seconds.

## The Vision (Next)

Once you can search your comms, you can analyze them:
- Track how often your team communicates during objective fights
- Surface the most critical moments automatically with LLM-powered analysis
- Visualize communication intensity across the game timeline
- Identify patterns in decision-making and shotcalling

The foundation is search. The goal is team communication intelligence.

**In competition, every slight advantage matters. Spend less time searching and more time improving.**

## Current Features (MVP 1.0)

**Session Management:**
- ✅ Create, list, edit, and delete sessions
- ✅ Track processing duration for each session

**Media Processing:**
- ✅ Upload audio (mp3, m4a, wav) or video (mp4, mkv, avi)
- ✅ Automatic audio extraction from video files (ffmpeg) with light denoising
- ✅ GPU-accelerated transcription (faster-whisper + CUDA)
- ✅ **League-specific term bank** (305 terms: champions, objectives, items, locations, mechanics)
- ✅ Post-normalization for common mishears (19 regex rules: apostrophe champions, abbreviations, etc.)
- ✅ Fuzzy post-processing with RapidFuzz (auto-corrects against term bank, score_cutoff=82)
- ✅ ~60% base accuracy on League terms, targeting 70-80% with fuzzy correction

**Search & Navigation:**
- ✅ **Keyword search** - Full-text search with BM25 ranking (SQLite FTS5), searches both transcript and notes
- ✅ **Time range filtering** - Filter chunks by game time (MM:SS format)
- ✅ **Embedded YouTube player** - Click timestamps to seek player instantly
- ✅ **Player position sync** - Active chunk highlights and auto-scrolls as video plays (toggleable)
- ✅ **Tab-based sidebar** - Switch between Sessions and Explore (transcript) views
- ✅ Pagination and chunk expansion (clickable text)
- ✅ Copy timestamp URL to clipboard
- ✅ **Chunk annotations** - Add coach notes to chunks (100 char limit, auto-save, searchable)

**UI/UX:**
- ✅ **Dark theme** - Sleek purple-accented design inspired by Linear/Vercel
- ✅ Chronological chunk flow with minimal visual separation
- ✅ Maximized player size for screensharing/presentation

**In-progress:**
- ⚙️ Chunk inline editing (fix transcription errors)
- ⚙️ Bookmarks/favorites (star important chunks)
- ⚙️ Demo video showing the app and its features at the end of MVP 1.0

See [PRODUCT.md](docs/PRODUCT.md) for the full roadmap (MVP 1.5, 2.0).

---

## Quick Start

### Prerequisites

**System dependencies:**
```bash
sudo apt update
sudo apt install -y ffmpeg sqlite3
```

**Optional: GPU acceleration (NVIDIA + CUDA)**
- WSL2: NVIDIA driver on Windows with WSL GPU support
- Inside WSL: CUDA Toolkit 12.x + cuDNN 9
```bash
sudo apt install -y cuda-toolkit-12-9 cudnn9-cuda-12
```

### Setup

**1. Clone and install backend:**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Create `.env` file in `backend/`:**
```bash
WHISPER_MODEL=small.en
TRANSCRIBE_DEVICE=cuda  # or 'cpu' if no GPU
DISABLE_LOL_NORMALIZE=0
DISABLE_FUZZY_CORRECT=0
DISABLE_AUDIO_DENOISE=0
```

**3. Start backend:**
```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

**4. Install and start frontend:**
```bash
cd frontend
npm install
npm run dev
```
Frontend: http://localhost:5173

---

## Usage

1. **Create a session** - Give it a title and YouTube URL for playback
2. **Upload audio/video** - Drop your scrim recording (or extract from OBS/Discord)
3. **Process** - Click "Process" to start transcription (~5 min per hour of audio)
4. **Explore** - Switch to Explore tab to view the searchable transcript
5. **Search** - Enter keywords like "Baron", "dive mid", "reset"
6. **Filter by time** - Enter start/end time (e.g., "5:00" to "10:00") to narrow results
7. **Seek** - Click chunk timestamps to seek the embedded YouTube player instantly

---

## Technical Stack

- **Backend:** FastAPI (Python), SQLite with FTS5, faster-whisper, ffmpeg, RapidFuzz
- **Frontend:** React + Vite
- **Environment:** WSL2, GPU-accelerated (CUDA 12.9), local-first (no cloud)
- **Data:** All sessions/chunks/audio stored locally in `backend/data/`

---

## Transcription Accuracy

**Current performance (small.en + term bank + fuzzy correction):**
- ~60% base accuracy on League-specific terms, targeting 70-80% with post-processing
- ~5 minutes processing per hour of audio (GPU)
- 19 regex rules auto-correct common mishears (harold→herald, word→ward, apostrophe champions, etc.)
- RapidFuzz fuzzy matching corrects close-spelling errors against term bank (score_cutoff=82)
- Light audio denoising via ffmpeg (highpass + lowpass + noise reduction)

**Planned improvements:**
- Fine-tuned model (MVP 2.0) → 80-90% accuracy

**Why not 100%?**
- Overlapping speech (teamfights)
- Diverse accents and speaking styles
- Non-standard terminology ("Nashie", "Eldorado")
- Background noise and audio quality

---

## Project Structure

```
vodcomms/
├── backend/
│   ├── main.py              # FastAPI app (sessions, transcription, search)
│   ├── term_bank.json       # League-specific vocabulary (305 terms)
│   ├── requirements.txt     # Python dependencies
│   ├── data/                # SQLite DB + uploaded media
│   └── .env.example         # Environment variables template
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component (~913 lines)
│   │   └── App.css          # Dark theme styling with purple accents
│   ├── package.json         # Node dependencies
│   └── vite.config.js       # Vite configuration
└── docs/
    └── PRODUCT.md           # Full product roadmap and vision
```

---

## Roadmap

**MVP 1.0 (Current):** Complete local tool with search, transcription, and time filtering
**MVP 1.5 (Next):** Speaker diarization, LLM-powered highlights, semantic search, multi-session search
**MVP 2.0 (Future):** Production SaaS with teams, cloud hosting, fine-tuned models, live transcription

See [PRODUCT.md](docs/PRODUCT.md) for detailed roadmap and feature priorities.

---

## Acknowledgments

- **faster-whisper** for GPU-accelerated transcription
- **SQLite FTS5** for blazing-fast full-text search
- **FastAPI** for the clean Python backend
- **React + Vite** for the snappy frontend
