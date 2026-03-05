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

## Demo

Watch the 4-minute walkthrough:

[![Recall.gg Demo](https://img.youtube.com/vi/5MuJDA3pIvg/0.jpg)](https://www.youtube.com/watch?v=5MuJDA3pIvg)

## The Vision (Next)

Once you can search your comms, you can analyze them:
- **Auto-generated session scorecards and chunk tags** via Claude Haiku (implemented)
- Track how often your team communicates during objective fights
- Visualize communication intensity across the game timeline
- Identify patterns in decision-making and shotcalling

Search is the foundation. The goal is to extract insights from team communication patterns.

**In competition, every slight advantage matters. Spend less time searching and more time improving.**

## Current Features (MVP 1.0)

**Session Management:**
- ✅ Create, list, edit, and delete sessions
- ✅ Track processing duration for each session

**Media Processing:**
- ✅ Upload audio (mp3, m4a, wav) or video (mp4, mkv, avi)
- ✅ Automatic audio extraction from video files (ffmpeg) with light denoising
- ✅ GPU-accelerated transcription (whisperX + CUDA, large-v3/int8) with word-level alignment
- ✅ **Speaker diarization** via pyannote (identifies who said what)
- ✅ **Speaker assignment** - Map speaker labels to real player names via modal with play-sample UX
- ✅ **League-specific term bank** (305 terms: champions, objectives, items, locations, mechanics)
- ✅ Post-normalization for common mishears (19 regex rules: apostrophe champions, abbreviations, etc.)
- ✅ Fuzzy post-processing with RapidFuzz (auto-corrects against term bank, score_cutoff=82)
- ✅ ~60% base accuracy on League terms, targeting 70-80% with fuzzy correction

**LLM Analysis (Claude Haiku):**
- ✅ **Session scorecard** - Five comms categories rated 1-10 (summoner tracking, objective setup, teamfight comms, shotcall clarity, map awareness)
- ✅ **Per-chunk tags** - Auto-tags notable moments: objective calls, shotcalls, disagreements, silences, tilt
- ✅ Fault-tolerant: failures never block processing (controlled by ANTHROPIC_API_KEY + DISABLE_LLM_ANALYSIS)

**Search & Navigation:**
- ✅ **Keyword search** - Postgres full-text search (tsvector + GIN), searches both transcript and notes
- ✅ **Semantic search** - Vector similarity via sentence-transformers + pgvector (hybrid with keyword search)
- ✅ **Time range filtering** - Filter chunks by game time (MM:SS format)
- ✅ **Bookmark filtering** - Filter to show only bookmarked chunks via "Show bookmarked only" toggle
- ✅ **Multi-session search** - Search across all sessions at once; results grouped by session with headers
- ✅ **Full-width Explore view** - Centered layout (max-width 900px) with prominent search bar
  - Session filter dropdown (search all or filter to one session)
  - Search term highlighting (purple marks on matched terms)
  - LLM tag dots + colored speaker dots on each result
  - Cross-session click-through: click a result → enters Session Mode with video auto-seek
  - Back button returns to Explore with search state preserved
- ✅ **Embedded YouTube player** - Click timestamps to seek player instantly; auto-switches videos across sessions
- ✅ **Player position sync** - Active chunk highlights and auto-scrolls as video plays (toggleable)
- ✅ **Session timeline** - Speaker activity visualization below YouTube player (recharts)
  - Per-speaker activity lines at 15-second intervals with 60-second sliding window
  - LLM tag markers (colored dots for objective calls, shotcalls, disagreements, silence, tilt)
  - Playhead synced with YouTube player, click-to-seek on timeline
- ✅ **Two-mode layout** - Navigation Mode (sidebar + main area) vs Session Mode (transcript panel + video/timeline)
- ✅ Pagination and chunk expansion (clickable text)

**Chunk Interaction:**
- ✅ **Action buttons** - Four per chunk (copy 📋, note 📝/✏️, bookmark ⭐/☆, expand ▼/▲)
- ✅ **Copy timestamp URL** to clipboard with 2-second confirmation
- ✅ **Bookmarks/favorites** - Star important chunks for quick access, searchable via filter toggle
- ✅ **Chunk inline editing** - Edit transcribed text directly (1000 char limit, auto-save on blur/Enter)
- ✅ **Chunk annotations** - Add coach notes to chunks (100 char limit, auto-save, searchable via full-text search)

**UI/UX:**
- ✅ **Dark theme** with purple accents (Linear/Vercel aesthetic)
- ✅ **Breadcrumbs** - "Sessions / [Session Name]" navigation bar in Session Mode
- ✅ **Mobile disclaimer** - Full-screen message for viewports below 768px
- ✅ **View transitions** - Smooth fadeIn/fadeSlideIn on view changes, scaleIn on modal open
- ✅ **Progressive disclosure** - Status badges, relative timestamps, collapsible panels
- ✅ **Two-mode layout** - Navigation Mode (sidebar + main) vs Session Mode (transcript + video)
- ✅ Chronological chunk ordering
- ✅ Large embedded player optimized for screen sharing

**MVP 1.0 Status:** ✅ Complete and ready for demo

See [PRODUCT.md](docs/PRODUCT.md) for the full roadmap (MVP 1.5, 2.0).

---

## Quick Start

### Prerequisites

**System dependencies:**
```bash
sudo apt update
sudo apt install -y ffmpeg docker.io docker-compose-v2
```

**Optional: GPU acceleration (NVIDIA + CUDA)**
- WSL2: NVIDIA driver on Windows with WSL GPU support
- Inside WSL: CUDA Toolkit 12.x + cuDNN 9
```bash
sudo apt install -y cuda-toolkit-12-9 cudnn9-cuda-12
```

### Setup

**1. Start Postgres:**
```bash
docker compose up -d
# Postgres at localhost:5432 (user=recall, password=recall, db=recall)
```

**2. Clone and install backend:**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Create `.env` file in `backend/`:**
```bash
WHISPER_MODEL=large-v3
TRANSCRIBE_DEVICE=cuda  # or 'cpu' if no GPU
WHISPER_COMPUTE_TYPE=int8  # int8 for large-v3 on 8GB GPUs, float16 for >=12GB
WHISPERX_BATCH_SIZE=4  # 4 for large-v3/int8 on 8GB, 16 for small.en
DISABLE_LOL_NORMALIZE=0
DISABLE_FUZZY_CORRECT=0
DISABLE_AUDIO_DENOISE=0
DISABLE_SEMANTIC_SEARCH=0
SEMANTIC_SEARCH_THRESHOLD=0.5
DISABLE_LLM_ANALYSIS=0
ANTHROPIC_API_KEY=  # Anthropic API key for Claude Haiku session analysis (empty = skip)
HF_TOKEN=hf_your_token_here  # HuggingFace token for speaker diarization
# Postgres defaults (localhost:5432/recall) match docker-compose.yml
```

**4. Start backend:**
```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

**5. Install and start frontend:**
```bash
cd frontend
npm install
npm run dev
```
Frontend: http://localhost:5173

---

## Usage

1. **Create a session** - Give it a title and YouTube URL for playback
2. **Upload audio/video** - Upload your scrim recording
3. **Process** - Click "Process" to start transcription (~5 min per hour of audio)
4. **Explore** - Switch to Explore tab to view the searchable transcript
5. **Search** - Enter keywords like "Baron", "dive mid", "reset"
6. **Filter by time** - Enter start/end time (e.g., "5:00" to "10:00") to narrow results
7. **Seek** - Click chunk timestamps to seek the embedded YouTube player instantly

---

## Technical Stack

- **Backend:** FastAPI (Python), Postgres with tsvector full-text search + pgvector semantic search, psycopg2, whisperX (transcription + alignment + diarization), sentence-transformers, Claude Haiku (LLM analysis), ffmpeg, RapidFuzz
- **Frontend:** React + Vite, recharts (timeline visualization)
- **Database:** Postgres (via Docker Compose) with pgvector extension, psycopg2 connection pool, no ORM
- **Environment:** WSL2, GPU-accelerated (CUDA 12.9), local-first (no cloud)
- **Data:** Sessions/chunks in Postgres, uploaded media/audio in `backend/data/`

---

## Transcription Accuracy

**Current performance (large-v3/int8 + term bank + fuzzy correction):**
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
│   ├── main.py              # Entry point for uvicorn
│   ├── app/
│   │   ├── main.py          # FastAPI app initialization
│   │   ├── config.py        # Environment config and constants
│   │   ├── models.py        # Pydantic request/response schemas
│   │   ├── database.py      # Postgres connection pool and queries
│   │   ├── routers/         # API endpoints (sessions, chunks, media)
│   │   └── services/        # Business logic (audio, transcription, text_processing, embedding, llm)
│   ├── term_bank.json       # League-specific vocabulary (305 terms)
│   ├── scripts/             # Utility scripts (backfill_embeddings)
│   ├── requirements.txt     # Python dependencies
│   ├── data/                # Uploaded media + extracted audio
│   └── .env.example         # Environment variables template
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main component, view routing, cross-view navigation
│   │   ├── components/      # React components (Sidebar, SessionList, SessionDetail, TranscriptPanel, ExplorePanel, YouTubePlayer, SessionTimeline, Breadcrumbs, MobileDisclaimer)
│   │   ├── hooks/           # Custom hooks (useAppState, useYouTubePlayer)
│   │   ├── utils/           # Utilities (formatters, api client)
│   │   └── styles/          # Global CSS (variables, base styles)
│   ├── package.json         # Node dependencies
│   └── vite.config.js       # Vite configuration
└── docs/
    └── PRODUCT.md           # Full product roadmap and vision
```

---

## Roadmap

**MVP 1.0 (✅ Complete):** Full local tool with search, transcription, time filtering, bookmarks, annotations, and LLM analysis
**MVP 1.5 (In Progress):** Speaker assignment (done), full-width Explore view (done), speaker filtering, intensity detection, session comparison
**MVP 2.0 (Future):** Production SaaS with teams, cloud hosting, fine-tuned models, live transcription

See [PRODUCT.md](docs/PRODUCT.md) for detailed roadmap and feature priorities.

---

## Acknowledgments

- **whisperX** (and faster-whisper, pyannote.audio)
- **PostgreSQL**
- **FastAPI**
- **React + Vite**
- **RapidFuzz**
- **sentence-transformers** + **pgvector**
- **Anthropic Claude Haiku**
