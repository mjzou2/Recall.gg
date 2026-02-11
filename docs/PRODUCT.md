# RECALL.GG
Local-first esports VOD comms search and decision archaeology tool.

## One-liner
Turn hours of scrim comms into a searchable, timestamped knowledge base. Find any moment, decision, or callout in seconds.

## The wedge (why this is different)
Most VOD tools parse **game events** (kills, objectives, gold). RECALL.GG parses **human decisions** from voice comms:
- Plan formation: "we should Baron after this wave"
- Plan evolution: "no wait, they have TP, let's reset"
- Conflict/uncertainty: overlapping voices, disagreements, hesitation
- Attribution: "I called the dive", "my bad, I didn't see"

**Goal:** Make team review faster, more objective, and focused on decision-making rather than stats.

---

## MVP 1.0 (Complete Local Tool)
**Status: In Progress → Target: Demo-ready in 2-3 days**

### Core Features (✓ = Implemented, ⚙️ = In Progress)

**Session Management:**
- ✓ Create, list, edit, delete sessions
- ✓ Session notes field (add context/description)
- ✓ Processing duration tracking

**Media Processing:**
- ✓ Upload audio (mp3/m4a/wav) and video (mp4/mkv/avi)
- ✓ Audio extraction via ffmpeg with light denoising (highpass + lowpass + afftdn)
- ✓ Transcription pipeline:
  - ✓ faster-whisper with GPU support (CUDA)
  - ✓ League-specific term bank (300+ terms: champions, objectives, comms vocabulary)
  - ✓ Contextual initial_prompt + hotwords for domain accuracy
  - ✓ Model selection (base/small/medium) - currently using small.en
  - ✓ Post-normalization for common mishears (19 regex rules: apostrophe champions, abbreviations, etc.)
  - ✓ **Fuzzy post-processing with RapidFuzz** - auto-corrects against term bank (score_cutoff=82)
  - ✓ Achieves ~60% base accuracy on League terms, ~5 min per hour of audio
  - ✓ Target with fuzzy processing: 70-80% accuracy
- ✓ Chunk storage with timestamps in SQLite

**Search & Navigation:**
- ✓ Keyword search with FTS5 + BM25 ranking
- ✓ Time range filtering (filter chunks by MM:SS game time)
- ⚙️ **Synonym expansion** - Hardcoded alias map for League-specific terms
  - e.g. searching "baron" also matches "Nashor", "Nash"
  - e.g. searching "herald" also matches "Shelly"
  - Expands FTS query tokens before hitting SQLite, no new dependencies
- ✓ Clickable timestamps (open YouTube URL at exact moment)
- ✓ Copy timestamp to clipboard

**Playback:**
- ⚙️ **Embedded YouTube player with bidirectional sync**
  - Click chunk → player jumps to timestamp
  - Player reaches chunk → chunk highlights in list
  - Seamless in-app playback, no tab switching

**Chunk Interaction:**
- ⚙️ **Manual chunk annotations** - Add coach notes to specific chunks
  - "Game-losing call", "Review with ADC", "Good macro example"
  - Stored as notes field on chunks table
- ⚙️ **Chunk inline editing** - Fix transcription errors directly
  - Click to edit → updates chunk text → improves future search
- ⚙️ **Bookmarks/favorites** - Star important chunks for quick access
  - Filter view: "Show me all starred chunks from this session"

### Technical Stack
- **Backend:** FastAPI (Python), SQLite with FTS5, faster-whisper, ffmpeg, RapidFuzz
- **Frontend:** React + Vite, minimal styling (polish planned)
- **Environment:** WSL2, GPU-accelerated (CUDA 12.9), local-first (no cloud)
- **Data:** All sessions/chunks/audio stored locally in `backend/data/`

### Constraints
- **No YouTube download** - User provides YouTube URL for playback, uploads audio separately
- **Synchronous processing** - No task queue yet, transcription blocks until complete
- **Single user** - No authentication or multi-user support

---

## MVP 1.5 (Intelligence Layer)
**Goal:** Add ML/LLM-powered features for deeper insights.

### Speaker Intelligence
- **Speaker diarization** - Identify who said what (Top, JG, Mid, ADC, Supp)
  - Use pyannote.audio for voice clustering
  - Manual labeling fallback (one-time setup per team)
  - Target: 65-75% accuracy (good enough with manual corrections)
  - Enables searches like "show me everything the jungler said about Baron"

### Content Intelligence
- **Intensity detection** - Flag high-energy moments (teamfights, conflicts)
  - Audio features: volume spikes, speech rate, overlapping voices
  - Auto-tag as "high intensity" for quick review
- **Top moments (LLM-powered)** - Auto-generate 5-10 key timestamps per session
  - LLM scans chunks, identifies decisions/mistakes/highlights
  - Saves coaches from reading 200+ chunks manually
- **Auto-tagging** - LLM classifies chunks by type
  - Categories: fight, macro, draft, lane, decision, mistake
  - Enables filtered views: "show me all macro discussion"

### Advanced Search
- **Semantic search** - Search by concept, not just keywords
  - Use embeddings to find "macro mistakes" or "shotcalling confusion"
  - Handles synonyms and related concepts
- **Multi-session search** - "Find all Baron calls across last 5 scrims"
- **Session comparison** - Compare themes/patterns across multiple sessions
- **Saved searches** - Store frequent queries for one-click access

### Pattern Recognition
- **Recurring issues tracking** - Surface repeated problems
  - "You've mentioned 'no vision bot' in 8/10 scrims"
  - Helps teams identify systemic weaknesses
- **Decision timeline view** - Visualize plan evolution
  - Plan formation → changes → commitment → outcome

### Export & Sharing
- **Export timestamps** - Generate shareable timestamp list (text/CSV/JSON)
- **Clip creation** - Extract 30-60s clips around key moments
  - Use ffmpeg to cut video segments
  - Share individual moments without full VOD

---

## MVP 2.0 (Production SaaS)
**Goal:** Turn this into a hosted product teams can pay for.

### Authentication & Teams
- **User accounts** - Sign up, login, session management
- **Team workspaces** - Shared sessions, role-based permissions
- **Coach/player views** - Coaches see all, players see filtered content

### Cloud Infrastructure
- **Hosted option** - Cloud transcription for users without GPUs
- **Hybrid mode** - Users choose local (free, private) or cloud (paid, convenient)
- **BYO API key** - Users provide their own Whisper API key for cloud processing
- **Async task queue** - Celery + Redis for non-blocking transcription

### Advanced Features
- **Fine-tuned models** - League-specific Whisper trained on labeled comms
  - Target: 80-90% accuracy on domain terms
  - Requires collecting 5-10 hours of transcribed audio
- **Live transcription** - Real-time searchable comms during scrims
  - Desktop app or OBS plugin to capture Discord audio
  - Incremental transcription with streaming display
- **Timeline visualization** - Visual timeline showing:
  - When chunks occurred (bars on timeline)
  - Intensity/density (talking vs silence)
  - Filtered results highlighted
  - Think: YouTube chapter markers but for comms
- **Keyboard shortcuts** - Speed up workflow during live review
  - Space = play/pause, ←/→ = skip 5s, J/K = prev/next chunk
  - F = favorite chunk, N = add note
- **Bulk operations** - Select multiple chunks and:
  - Tag them all at once
  - Export selected chunks only
  - Delete/hide irrelevant chunks
- **Integrations** - Discord bot, Slack notifications, calendar sync

---

## Success Criteria (What "Good" Looks Like)

**For MVP 1.0:**
- A coach can search "Baron" and find every mention across a scrim in <5 seconds
- Transcription accuracy is 70-80% on League terms (good enough for reliable search)
- Coaches save 20+ minutes per review session vs scrubbing through full VOD
- Coaches can annotate key moments and fix errors inline
- Embedded player makes review feel integrated, not clunky

**For MVP 1.5:**
- Coaches can identify recurring issues across multiple scrims without manual note-taking
- Speaker attribution works well enough that "show me everything the jungler said" is useful
- LLM highlights save coaches from reading every chunk
- Semantic search finds moments even when exact keywords weren't used

**For MVP 2.0:**
- Teams pay for the product (validation: solves real problem)
- 80%+ accuracy on League terms with fine-tuned model
- Review sessions become data-driven: "we called Baron 12 times, succeeded 3 times"
- Live transcription enables real-time feedback during scrims

---

## Development Roadmap

### Immediate Next Steps (MVP 1.0 Completion - 2-3 days)
1. ⚙️ **Embedded YouTube player with bidirectional sync** (4-6 hours)
2. ⚙️ **Manual chunk annotations** - notes field (2-3 hours)
3. ⚙️ **Chunk inline editing** - fix transcription errors (2-3 hours)
4. ⚙️ **Bookmarks/favorites** - star chunks (1-2 hours)
5. ✓ **Fuzzy post-processing** - RapidFuzz correction
6. ⚙️ **Synonym expansion** - hardcoded alias map for search (1-2 hours)
7. ⚙️ **Demo preparation** - record video, write README, polish UI (2-4 hours)

**Total estimated: 12-20 hours → Target: Demo-ready by end of week**

### Post-Demo Priorities (Driven by User Feedback)
- Code refactor (split main.py and App.jsx into modules)
- Frontend polish (component library, dark mode, consistent styling)
- Real-time progress bar for transcription
- Evaluation harness for measuring transcription quality improvements

### MVP 1.5 Feature Priority (Based on Coach Feedback)
1. Speaker diarization (if coaches say "can't tell who said what")
2. Top moments LLM (if coaches say "too many chunks to review")
3. Semantic search (if coaches say "keyword search misses things")
4. Timeline visualization (if coaches say "hard to see patterns")
5. Intensity detection (if coaches say "need to find teamfight comms faster")

---

## Known Limitations

**Transcription Accuracy:**
- Base Whisper model: ~30% accuracy on League terms
- Current (small.en + term bank + fuzzy): ~60% base, targeting 70-80% with post-processing
- Overlapping speech and diverse accents reduce accuracy further
- Fine-tuning could reach 80-90% but requires significant data collection effort

**Architecture:**
- Synchronous processing blocks the API (5 min per hour of audio)
- No concurrent session processing
- Processing progress shown as elapsed timer only (no % complete)

**Scope:**
- YouTube-only for video playback (no local video player yet)
- Manual audio upload (no YouTube download automation due to ToS)
- Single-us