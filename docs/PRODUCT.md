# RECALL.GG (Working Name)
Esports scrim/VOD comms search + “decision archaeology” tool.

## One-liner
Turn messy scrim comms into a searchable timeline so coaches/players can instantly answer: **what did we decide, when, and why?**

## The wedge (why this is different)
Most VOD tools index **game events** (baron/dragon/kills). RECALL.GG indexes **intent + decisions** from comms:
- plan formation (“we should…”, “next we…”, “play for…”)
- plan changes (“no, don’t”, “we can’t”, “reset”, “turn”)
- disagreement/hesitation moments
- responsibility (“I called…”, “my bad…”, “I thought…”)

Goal: make review faster, clearer, and less argumentative.

## MVP 1.0 (local-first)
**Inputs**
- Local audio/video upload (mp3/m4a/wav/mp4)
- Optional YouTube URL (for playback + timestamp jump; app does NOT download from YouTube)

**Pipeline**
1) Extract/standardize audio (ffmpeg)
2) Transcribe with timestamps (faster-whisper)
3) Chunk transcript into time windows
4) Store in SQLite
5) Search (keyword + semantic)
6) Click result → jump to timestamp (YouTube link or local player)

**UI**
- Session list + status
- Session viewer: player + search + results + timeline

**Non-goals**
- Auth/teams/workspaces
- Cloud hosting
- Speaker diarization (who spoke)
- Auto-highlights/tags/summaries
- Game event ingestion (baron timers, etc.)

## MVP 1.5 (make it “coach-worthy”)
- Saved searches + shareable cliplists (timestamps)
- Notes + tags per moment
- Auto-summary per session (LLM-assisted)
- “Decision timeline” view (plan → change → commit)
- Disagreement/hesitation detector (simple heuristics at first)

## MVP 2.0 (SaaS: bring-your-own-VOD)
- Login + team workspace
- Store transcripts/embeddings/notes; VOD remains external link
- Usage-based ingestion options (local vs API) and/or BYO API key

## Success criteria (what “good” looks like)
- A coach can find “every time we said reset/baron/turn” across scrims in seconds
- Review becomes faster: fewer minutes scrubbing, more time coaching
- Disputes resolved quickly: “who called wh
## Run + verify transcription
1) Backend: `cd backend && source .venv/bin/activate && pip install faster-whisper && python -m uvicorn main:app --reload --port 8000`
2) Create a session: `curl -X POST http://localhost:8000/sessions -H "Content-Type: application/json" -d '{}'`
3) Upload media: `curl -F "file=@/full/path/to/audio-or-video" http://localhost:8000/sessions/<session_id>/media`
4) Process + transcribe: `curl -X POST http://localhost:8000/sessions/<session_id>/process`
5) Verify output: `curl http://localhost:8000/sessions/<session_id>` should show `audio_path` set and `chunks` populated (start/end ms + text). Launch the frontend to confirm chunks render in the UI.
::contentReference[oaicite:0]{index=0}
