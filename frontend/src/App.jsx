import { useEffect, useState } from 'react'
import './App.css'
import SessionDetails from './components/SessionDetails'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

const formatTime = (ms) => {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

function App() {
  const [title, setTitle] = useState('')
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [sessions, setSessions] = useState([])
  const [sessionId, setSessionId] = useState('')
  const [sessionDetails, setSessionDetails] = useState(null)
  const [file, setFile] = useState(null)
  const [chunks, setChunks] = useState([])
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const [isCreating, setIsCreating] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [isSearching, setIsSearching] = useState(false)
  const [lastQuery, setLastQuery] = useState('')

  useEffect(() => {
    loadSessions()
  }, [])

  const loadSessions = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`)
      if (!res.ok) throw new Error('Failed to load sessions')
      const data = await res.json()
      setSessions(data)
    } catch (err) {
      setError(err.message)
    }
  }

  const loadSessionDetails = async (id) => {
    setStatus('Loading session...')
    setError('')
    try {
      const res = await fetch(`${API_BASE}/sessions/${id}`)
      if (!res.ok) throw new Error('Session not found')
      const data = await res.json()
      setSessionDetails(data.session)
      setChunks(data.chunks || [])
      setSessionId(id)
      setStatus('Ready')
    } catch (err) {
      setError(err.message)
    }
  }

  const handleCreateSession = async (event) => {
    event.preventDefault()
    setIsCreating(true)
    setError('')
    setStatus('Creating session...')
    try {
      const res = await fetch(`${API_BASE}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: title || null,
          youtube_url: youtubeUrl || null,
        }),
      })
      if (!res.ok) throw new Error('Failed to create session')
      const data = await res.json()
      setSessionId(data.id)
      setSessionDetails(data)
      setChunks([])
      setStatus('Session created')
      await loadSessions()
    } catch (err) {
      setError(err.message)
    } finally {
      setIsCreating(false)
    }
  }

  const handleUpload = async (event) => {
    event.preventDefault()
    if (!sessionId) {
      setError('Create or select a session first.')
      return
    }
    if (!file) {
      setError('Choose a media file to upload.')
      return
    }
    setIsUploading(true)
    setError('')
    setStatus('Uploading media...')
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${API_BASE}/sessions/${sessionId}/media`, {
        method: 'POST',
        body: form,
      })
      if (!res.ok) throw new Error('Upload failed')
      await loadSessionDetails(sessionId)
      setStatus('Media uploaded')
    } catch (err) {
      setError(err.message)
    } finally {
      setIsUploading(false)
    }
  }

  const handleProcess = async () => {
    if (!sessionId) {
      setError('Create or select a session first.')
      return
    }
    setIsProcessing(true)
    setError('')
    setStatus('Processing (dummy chunks)...')
    try {
      const res = await fetch(`${API_BASE}/sessions/${sessionId}/process`, {
        method: 'POST',
      })
      if (!res.ok) throw new Error('Process failed')
      const data = await res.json()
      setChunks(data.chunks || [])
      setStatus('Chunks ready')
      await loadSessionDetails(sessionId)
      await loadSessions()
    } catch (err) {
      setError(err.message)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleSearch = async () => {
    if (!sessionId) return
    const trimmed = searchQuery.trim()
    if (!trimmed) {
      setSearchQuery('')
      setIsSearching(false)
      setLastQuery('')
      await loadSessionDetails(sessionId)
      setStatus('Search cleared')
      return
    }
    setStatus('Searching...')
    setError('')
    try {
      const res = await fetch(`${API_BASE}/sessions/${sessionId}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: trimmed, limit: 50 }),
      })
      if (!res.ok) throw new Error('Search failed')
      const data = await res.json()
      setChunks(data.results || [])
      setIsSearching(true)
      setLastQuery(trimmed)
      setStatus(`Search returned ${data.results?.length ?? 0} results`)
    } catch (err) {
      setError(err.message)
    }
  }

  const handleClearSearch = async () => {
    setSearchQuery('')
    setIsSearching(false)
    setLastQuery('')
    if (sessionId) {
      await loadSessionDetails(sessionId)
    }
    setStatus('Search cleared')
  }

  return (
    <div className="page">
      <header>
        <div>
          <p className="eyebrow">RECALL.GG · local-first</p>
          <h1>Scrim transcripts in one place</h1>
          <p className="lede">
            Create a session, upload a VOD or audio file, and generate dummy
            chunks. We will swap in real transcription later.
          </p>
        </div>
        <div className="status">
          <span className="badge">{status || 'Idle'}</span>
          {error && <span className="badge danger">{error}</span>}
        </div>
      </header>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Step 1</p>
            <h2>Create a session</h2>
          </div>
          <button className="ghost" onClick={loadSessions}>
            Refresh sessions
          </button>
        </div>
        <form className="stack" onSubmit={handleCreateSession}>
          <label className="field">
            <span>Title</span>
            <input
              type="text"
              placeholder="Scrim vs Team Blue"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
          </label>
          <label className="field">
            <span>YouTube URL (stored only)</span>
            <input
              type="url"
              placeholder="https://youtube.com/watch?v=..."
              value={youtubeUrl}
              onChange={(e) => setYoutubeUrl(e.target.value)}
            />
          </label>
          <div className="actions">
            <button type="submit" disabled={isCreating}>
              {isCreating ? 'Creating...' : 'Create session'}
            </button>
            {sessionId && (
              <span className="hint">
                Active session: <code>{sessionId}</code>
              </span>
            )}
          </div>
        </form>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Step 2</p>
            <h2>Upload media</h2>
          </div>
          <span className="hint">
            Max: local files only. We do not download from YouTube.
          </span>
        </div>
        <form className="stack" onSubmit={handleUpload}>
          <label className="field file">
            <span>Choose video or audio</span>
            <input
              type="file"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>
          <div className="actions">
            <button type="submit" disabled={isUploading}>
              {isUploading ? 'Uploading...' : 'Upload to session'}
            </button>
            <button
              type="button"
              className="ghost"
              onClick={handleProcess}
              disabled={isProcessing || !sessionDetails?.media_path}
            >
              {isProcessing ? 'Processing...' : 'Process (dummy chunks)'}
            </button>
          </div>
        </form>
      </section>

      <section className="grid">
        <div className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Sessions</p>
              <h2>Pick or revisit</h2>
            </div>
          </div>
          <div className="session-list">
            {sessions.length === 0 && (
              <p className="hint">No sessions yet. Create one to begin.</p>
            )}
            {sessions.map((session) => (
              <button
                key={session.id}
                className={`session-card ${
                  session.id === sessionId ? 'active' : ''
                }`}
                onClick={() => loadSessionDetails(session.id)}
              >
                <div className="session-title">
                  <strong>{session.title || 'Untitled session'}</strong>
                  <span>{new Date(session.created_at).toLocaleString()}</span>
                </div>
                {session.youtube_url && (
                  <p className="hint">YouTube: {session.youtube_url}</p>
                )}
                {session.media_path && (
                  <p className="hint">Media stored at {session.media_path}</p>
                )}
              </button>
            ))}
          </div>
        </div>

        <SessionDetails session={sessionDetails} />

        {(sessionId || sessionDetails) && (
          <div className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Search</p>
                <h2>Keyword lookup</h2>
              </div>
            </div>
            <div className="stack">
              <input type="text" placeholder={'reset / baron / "we should"'} value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} />
              <div className="actions">
                <button type="button" onClick={handleSearch}>Search</button>
                <button type="button" className="ghost" onClick={handleClearSearch}>Clear</button>
              </div>
            </div>
          </div>
        )}

        <div className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Chunks</p>
              <h2>Dummy timeline</h2>
            </div>
          </div>
          {!sessionId && (
            <p className="hint">Select a session to view generated chunks.</p>
          )}
          {sessionId && chunks.length === 0 && (
            <p className="hint">No chunks yet. Process the uploaded file.</p>
          )}
          {sessionId && (
            <p className="hint">
              {isSearching
                ? `Results: ${chunks.length} (query: ${lastQuery})`
                : `Showing all chunks: ${chunks.length}`}
            </p>
          )}
          <div className="chunk-list">
            {chunks.map((chunk) => (
              <div key={chunk.id} className="chunk">
                <div className="chunk-times">
                  <span>{formatTime(chunk.start_ms)}</span>
                  <span>→</span>
                  <span>{formatTime(chunk.end_ms)}</span>
                </div>
                <p>{chunk.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}

export default App
