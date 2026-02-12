import { useEffect, useState, useRef } from 'react'
import './App.css'
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

const formatTime = (ms) => {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

const parseTime = (timeStr) => {
  if (!timeStr || !timeStr.trim()) return null
  const parts = timeStr.trim().split(':')
  if (parts.length !== 2) return null
  const minutes = parseInt(parts[0], 10)
  const seconds = parseInt(parts[1], 10)
  if (isNaN(minutes) || isNaN(seconds)) return null
  if (seconds < 0 || seconds >= 60) return null
  if (minutes < 0) return null
  return (minutes * 60 + seconds) * 1000
}

const formatDuration = (seconds) => {
  if (seconds == null) return null
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  if (mins === 0) return `${secs}s`
  return `${mins}m ${secs}s`
}

const shortenUrl = (url) => {
  if (!url) return ''
  try {
    const parsed = new URL(url)
    const tail = parsed.pathname.split('/').filter(Boolean).pop()
    const fragment = parsed.search || parsed.hash || ''
    const suffix = tail ? `${tail}${fragment}` : fragment.replace(/^\?/, '')
    return `${parsed.hostname}${suffix ? `/${suffix}` : ''}`
  } catch {
    const trimmed = url.replace(/^https?:\/\//, '')
    const parts = trimmed.split('/')
    if (parts.length <= 1) return trimmed
    const last = parts[parts.length - 1] || parts[parts.length - 2]
    return `${parts[0]}/${last}`
  }
}

const basename = (path) => {
  if (!path) return ''
  const normalized = path.split('?')[0]
  const parts = normalized.split('/')
  return parts[parts.length - 1] || normalized
}

const buildYoutubeUrlWithTimestamp = (baseUrl, startMs) => {
  if (!baseUrl) return null
  const seconds = Math.floor(startMs / 1000)
  try {
    const url = new URL(baseUrl)
    url.searchParams.set('t', `${seconds}s`)
    return url.toString()
  } catch {
    return `${baseUrl}${baseUrl.includes('?') ? '&' : '?'}t=${seconds}s`
  }
}

const extractYoutubeVideoId = (url) => {
  if (!url) return null
  try {
    const parsed = new URL(url)
    // Handle youtube.com/watch?v=VIDEO_ID
    if (parsed.hostname.includes('youtube.com') && parsed.searchParams.has('v')) {
      return parsed.searchParams.get('v')
    }
    // Handle youtu.be/VIDEO_ID
    if (parsed.hostname === 'youtu.be') {
      return parsed.pathname.slice(1) // Remove leading slash
    }
  } catch {
    // Invalid URL
  }
  return null
}

function App() {
  const [title, setTitle] = useState('')
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [notes, setNotes] = useState('')
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
  const [startTime, setStartTime] = useState('')
  const [endTime, setEndTime] = useState('')
  const [timeRangeError, setTimeRangeError] = useState('')
  const [lastTimeRange, setLastTimeRange] = useState('')
  const [pageIndex, setPageIndex] = useState(0)
  const [expandedChunkIds, setExpandedChunkIds] = useState(new Set())
  const [copiedChunkId, setCopiedChunkId] = useState(null)
  const [isEditingSession, setIsEditingSession] = useState(false)
  const [editTitle, setEditTitle] = useState('')
  const [editYoutubeUrl, setEditYoutubeUrl] = useState('')
  const [editNotes, setEditNotes] = useState('')
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [sidebarTab, setSidebarTab] = useState('sessions')
  const [youtubePlayer, setYoutubePlayer] = useState(null)
  const [isYoutubeApiReady, setIsYoutubeApiReady] = useState(false)
  const [activeChunkId, setActiveChunkId] = useState(null)
  const [autoScrollEnabled, setAutoScrollEnabled] = useState(true)

  const chunkListRef = useRef(null)
  const isAutoScrollingRef = useRef(false)
  const lastScrollTopRef = useRef(0)

  const pageSize = 25
  const canSearch = Boolean(sessionId)

  useEffect(() => {
    loadSessions()
  }, [])

  useEffect(() => {
    const totalPages = Math.max(1, Math.ceil(chunks.length / pageSize))
    if (pageIndex > totalPages - 1) {
      setPageIndex(0)
    }
  }, [chunks.length, pageIndex])

  // Timer for tracking elapsed processing time
  useEffect(() => {
    if (!isProcessing) {
      return
    }
    // Reset elapsed time when processing starts
    setElapsedSeconds(0)
    const interval = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1)
    }, 1000)
    // Cleanup: clear interval when processing ends or component unmounts
    return () => clearInterval(interval)
  }, [isProcessing])

  // Load YouTube iframe API
  useEffect(() => {
    // Check if API is already loaded
    if (window.YT && window.YT.Player) {
      setIsYoutubeApiReady(true)
      return
    }

    // Load the API script if not already present
    if (!window.YT) {
      const tag = document.createElement('script')
      tag.src = 'https://www.youtube.com/iframe_api'
      const firstScriptTag = document.getElementsByTagName('script')[0]
      firstScriptTag.parentNode.insertBefore(tag, firstScriptTag)
    }

    // Set up the callback for when API is ready
    window.onYouTubeIframeAPIReady = () => {
      setIsYoutubeApiReady(true)
    }
  }, [])

  // Initialize YouTube player when session loads with youtube_url
  useEffect(() => {
    // Clean up existing player
    if (youtubePlayer && youtubePlayer.destroy) {
      youtubePlayer.destroy()
      setYoutubePlayer(null)
    }

    // Check if we have everything needed to create a player
    if (!sessionDetails?.youtube_url || !isYoutubeApiReady) {
      return
    }

    const videoId = extractYoutubeVideoId(sessionDetails.youtube_url)
    if (!videoId) {
      console.warn('Could not extract video ID from:', sessionDetails.youtube_url)
      return
    }

    // Create the player
    const player = new window.YT.Player('youtube-player', {
      videoId: videoId,
      width: '100%',
      height: '100%',
      playerVars: {
        autoplay: 0,
        modestbranding: 1,
      },
      events: {
        onReady: (event) => {
          setYoutubePlayer(event.target)
        },
      },
    })

    // Cleanup on unmount or session change
    return () => {
      if (player && player.destroy) {
        player.destroy()
      }
    }
  }, [sessionDetails?.youtube_url, isYoutubeApiReady])

  // Player position sync - poll current time and highlight active chunk
  useEffect(() => {
    if (!youtubePlayer || !youtubePlayer.getCurrentTime || chunks.length === 0) {
      return
    }

    const interval = setInterval(() => {
      // Only update position when playing
      const playerState = youtubePlayer.getPlayerState?.()
      if (playerState !== window.YT?.PlayerState?.PLAYING) {
        return
      }

      const currentSeconds = youtubePlayer.getCurrentTime()
      const currentMs = Math.floor(currentSeconds * 1000)

      // Find the chunk that contains the current timestamp
      const activeChunk = chunks.find(
        (chunk) => chunk.start_ms <= currentMs && currentMs < chunk.end_ms
      )

      if (activeChunk) {
        const chunkKey =
          activeChunk.id ??
          `${activeChunk.start_ms}-${activeChunk.end_ms}-${activeChunk.text?.length ?? 0}`

        setActiveChunkId(chunkKey)

        // Auto-scroll to active chunk if enabled
        if (autoScrollEnabled) {
          // Find which page this chunk is on
          const chunkIndex = chunks.findIndex((chunk) => {
            const key =
              chunk.id ?? `${chunk.start_ms}-${chunk.end_ms}-${chunk.text?.length ?? 0}`
            return key === chunkKey
          })

          if (chunkIndex !== -1) {
            const targetPage = Math.floor(chunkIndex / pageSize)

            // If chunk is on a different page, switch to that page first
            if (targetPage !== pageIndex) {
              setPageIndex(targetPage)
              // Wait for React to re-render the new page before scrolling
              setTimeout(() => {
                isAutoScrollingRef.current = true
                const chunkElement = document.querySelector(
                  `[data-chunk-key="${chunkKey}"]`
                )
                if (chunkElement) {
                  chunkElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'nearest',
                  })
                }
                setTimeout(() => {
                  isAutoScrollingRef.current = false
                }, 1000)
              }, 100)
            } else {
              // Same page, scroll immediately
              isAutoScrollingRef.current = true
              const chunkElement = document.querySelector(
                `[data-chunk-key="${chunkKey}"]`
              )
              if (chunkElement) {
                chunkElement.scrollIntoView({
                  behavior: 'smooth',
                  block: 'nearest',
                })
              }
              setTimeout(() => {
                isAutoScrollingRef.current = false
              }, 1000)
            }
          }
        }
      }
    }, 500)

    return () => clearInterval(interval)
  }, [youtubePlayer, chunks, autoScrollEnabled])

  // Disable auto-scroll when user manually scrolls
  useEffect(() => {
    const container = chunkListRef.current
    if (!container) return

    // Initialize last scroll position
    lastScrollTopRef.current = container.scrollTop

    const handleScroll = () => {
      // Only disable if this is a user-initiated scroll, not a programmatic one
      if (!isAutoScrollingRef.current && autoScrollEnabled) {
        const scrollDistance = Math.abs(container.scrollTop - lastScrollTopRef.current)

        // Only disable if user scrolled more than 50 pixels (prevents accidental tiny scrolls)
        if (scrollDistance > 50) {
          setAutoScrollEnabled(false)
        }
      }

      // Update last scroll position
      lastScrollTopRef.current = container.scrollTop
    }

    container.addEventListener('scroll', handleScroll)
    return () => container.removeEventListener('scroll', handleScroll)
  }, [autoScrollEnabled])

  const resetChunkViews = () => {
    setPageIndex(0)
    setExpandedChunkIds(new Set())
  }

  const getPreviewText = (text) => {
    if (!text) return ''
    const firstLine = text.split('\n')[0]
    if (firstLine.length > 140) return `${firstLine.slice(0, 140)}...`
    if (text.length > firstLine.length) return `${firstLine}...`
    return firstLine
  }

  const toggleChunkExpanded = (chunkKey) => {
    setExpandedChunkIds((prev) => {
      const next = new Set(prev)
      if (next.has(chunkKey)) {
        next.delete(chunkKey)
      } else {
        next.add(chunkKey)
      }
      return next
    })
  }

  const handleCopyTimestamp = async (chunkKey, youtubeLink) => {
    if (!youtubeLink) return
    try {
      await navigator.clipboard.writeText(youtubeLink)
      setCopiedChunkId(chunkKey)
      setTimeout(() => setCopiedChunkId(null), 2000) // Clear after 2 seconds
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  const handleTimestampClick = (chunk) => {
    // Immediately highlight the clicked chunk
    const chunkKey =
      chunk.id ?? `${chunk.start_ms}-${chunk.end_ms}-${chunk.text?.length ?? 0}`
    setActiveChunkId(chunkKey)

    if (youtubePlayer && youtubePlayer.seekTo) {
      const seconds = Math.floor(chunk.start_ms / 1000)
      youtubePlayer.seekTo(seconds, true)
    } else {
      // Fallback to opening in new tab if player not available
      const youtubeLink = buildYoutubeUrlWithTimestamp(
        sessionDetails?.youtube_url,
        chunk.start_ms
      )
      if (youtubeLink) {
        window.open(youtubeLink, '_blank')
      }
    }
  }

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
      setIsEditingSession(false)
      setEditTitle('')
      setEditYoutubeUrl('')
      resetChunkViews()
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
          notes: notes || null,
        }),
      })
      if (!res.ok) throw new Error('Failed to create session')
      const data = await res.json()
      setSessionId(data.id)
      setSessionDetails(data)
      setChunks([])
      setIsEditingSession(false)
      setEditTitle('')
      setEditYoutubeUrl('')
      resetChunkViews()
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
    setStatus('Processing (transcribe + chunk)...')
    try {
      const res = await fetch(`${API_BASE}/sessions/${sessionId}/process`, {
        method: 'POST',
      })
      if (!res.ok) throw new Error('Process failed')
      const data = await res.json()
      setChunks(data.chunks || [])
      resetChunkViews()
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
    const startTimeInput = startTime.trim()
    const endTimeInput = endTime.trim()

    // If no filters provided, reset to all chunks
    if (!trimmed && !startTimeInput && !endTimeInput) {
      setSearchQuery('')
      setStartTime('')
      setEndTime('')
      setIsSearching(false)
      setLastQuery('')
      setLastTimeRange('')
      setTimeRangeError('')
      await loadSessionDetails(sessionId)
      setStatus('Search cleared')
      return
    }

    // Parse time inputs
    const startMs = startTimeInput ? parseTime(startTimeInput) : null
    const endMs = endTimeInput ? parseTime(endTimeInput) : null

    // Validate time range
    if (startTimeInput && startMs === null) {
      setTimeRangeError('Invalid start time format (use MM:SS)')
      return
    }
    if (endTimeInput && endMs === null) {
      setTimeRangeError('Invalid end time format (use MM:SS)')
      return
    }
    if (startMs !== null && endMs !== null && startMs >= endMs) {
      setTimeRangeError('End time must be after start time')
      return
    }

    // Clear any previous errors
    setTimeRangeError('')
    setStatus('Searching...')
    setError('')

    try {
      const res = await fetch(`${API_BASE}/sessions/${sessionId}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: trimmed,
          limit: 50,
          start_time_ms: startMs,
          end_time_ms: endMs,
        }),
      })
      if (!res.ok) throw new Error('Search failed')
      const data = await res.json()
      setChunks(data.results || [])
      setIsSearching(true)
      setLastQuery(trimmed)

      // Track time range for display
      if (startTimeInput || endTimeInput) {
        setLastTimeRange(`${startTimeInput || '0:00'}-${endTimeInput || '∞'}`)
      } else {
        setLastTimeRange('')
      }

      resetChunkViews()
      setStatus(`Search returned ${data.results?.length ?? 0} results`)
    } catch (err) {
      setError(err.message)
    }
  }

  const handleClearSearch = async () => {
    setSearchQuery('')
    setStartTime('')
    setEndTime('')
    setTimeRangeError('')
    setIsSearching(false)
    setLastQuery('')
    setLastTimeRange('')
    resetChunkViews()
    if (sessionId) {
      await loadSessionDetails(sessionId)
    }
    setStatus('Search cleared')
  }

  const handleEditSession = () => {
    if (!sessionDetails) return
    setEditTitle(sessionDetails.title || '')
    setEditYoutubeUrl(sessionDetails.youtube_url || '')
    setEditNotes(sessionDetails.notes || '')
    setIsEditingSession(true)
  }

  const handleCancelEdit = () => {
    setIsEditingSession(false)
    setEditTitle('')
    setEditYoutubeUrl('')
    setEditNotes('')
  }

  const handleSaveSession = async () => {
    if (!sessionId) return
    setStatus('Saving session...')
    setError('')
    try {
      const res = await fetch(`${API_BASE}/sessions/${sessionId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: editTitle.trim() || null,
          youtube_url: editYoutubeUrl.trim() || null,
          notes: editNotes.trim() || null,
        }),
      })
      if (!res.ok) throw new Error('Failed to update session')
      const data = await res.json()
      setSessionDetails(data)
      setIsEditingSession(false)
      await loadSessions()
      setStatus('Session updated')
    } catch (err) {
      setError(err.message)
    }
  }

  const handleDeleteSession = async () => {
    if (!sessionId) return
    const confirmed = window.confirm(
      'Delete this session? This will remove all chunks and media for it.'
    )
    if (!confirmed) return
    setStatus('Deleting session...')
    setError('')
    try {
      const res = await fetch(`${API_BASE}/sessions/${sessionId}`, {
        method: 'DELETE',
      })
      if (!res.ok) throw new Error('Failed to delete session')
      setSessionId('')
      setSessionDetails(null)
      setChunks([])
      setSearchQuery('')
      setIsSearching(false)
      setLastQuery('')
      resetChunkViews()
      setIsEditingSession(false)
      await loadSessions()
      setStatus('Session deleted')
    } catch (err) {
      setError(err.message)
    }
  }

  const totalPages = Math.max(1, Math.ceil(chunks.length / pageSize))
  const pageStart = pageIndex * pageSize
  const pageChunks = chunks.slice(pageStart, pageStart + pageSize)

  return (
    <div className="page">
      <header>
        <div>
          <p className="eyebrow">Scrim reviews made efficient</p>
          <h1>RECALL.GG</h1>
          <p className="lede">
            Create a session, upload a VOD or audio file, and generate
            chunks.
          </p>
        </div>
        <div className="status">
          <span className="badge">{status || 'Idle'}</span>
          {error && <span className="badge danger">{error}</span>}
        </div>
      </header>

      <section className="main-layout">
        <div className="left-column">
          <div className="sidebar-tabs">
            <button
              className={sidebarTab === 'sessions' ? 'active' : ''}
              onClick={() => setSidebarTab('sessions')}
            >
              Sessions
            </button>
            <button
              className={sidebarTab === 'explore' ? 'active' : ''}
              onClick={() => setSidebarTab('explore')}
            >
              Explore
            </button>
          </div>

          <div className="sidebar-content">
          {sidebarTab === 'sessions' && (
          <div className="sessions-tab">
          <div className="left-scroll">
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
                  <span>YouTube URL</span>
                  <input
                    type="url"
                    placeholder="https://youtube.com/watch?v=..."
                    value={youtubeUrl}
                    onChange={(e) => setYoutubeUrl(e.target.value)}
                  />
                </label>
                <label className="field">
                  <span>
                    Notes <span className="hint">(optional, {150 - notes.length} chars left)</span>
                  </span>
                  <textarea
                    placeholder="e.g., 'Scrim vs Team Red - Baron control focus'"
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    maxLength={150}
                    rows={2}
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
                  Max: local files only.
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
                    {isProcessing
                      ? `Processing... ${formatTime(elapsedSeconds * 1000)}`
                      : 'Process (transcribe + chunk)'}
                  </button>
                </div>
              </form>
            </section>

            <section className="panel">
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
                      <p className="hint" title={session.youtube_url}>
                        YouTube: {shortenUrl(session.youtube_url)}
                      </p>
                    )}
                    {session.media_path && (
                      <p className="hint" title={session.media_path}>
                        Media: {basename(session.media_path)}
                      </p>
                    )}
                    {session.notes && (
                      <p className="session-notes-preview">
                        {session.notes.length > 60
                          ? session.notes.slice(0, 60) + '...'
                          : session.notes}
                      </p>
                    )}
                  </button>
                ))}
              </div>
            </section>

          </div>

          <div className="left-footer">
            {sessionDetails && (
              <section className="panel">
                <div className="panel-header">
                  <div>
                    <p className="eyebrow">Selected</p>
                    <h2>Session details</h2>
                  </div>
                  <div className="actions">
                    {isEditingSession ? (
                      <>
                        <button type="button" onClick={handleSaveSession}>
                          Save
                        </button>
                        <button
                          type="button"
                          className="ghost"
                          onClick={handleCancelEdit}
                        >
                          Cancel
                        </button>
                      </>
                    ) : (
                      <>
                        <button
                          type="button"
                          onClick={handleDeleteSession}
                        >
                          Delete
                        </button>
                        <button
                          type="button"
                          className="ghost"
                          onClick={handleEditSession}
                        >
                          Edit
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {isEditingSession ? (
                  <div className="stack">
                    <label className="field">
                      <span>Title</span>
                      <input
                        type="text"
                        value={editTitle}
                        onChange={(e) => setEditTitle(e.target.value)}
                      />
                    </label>
                    <label className="field">
                      <span>YouTube URL</span>
                      <input
                        type="url"
                        value={editYoutubeUrl}
                        onChange={(e) => setEditYoutubeUrl(e.target.value)}
                      />
                    </label>
                    <label className="field">
                      <span>
                        Notes <span className="hint">({150 - editNotes.length} chars left)</span>
                      </span>
                      <textarea
                        value={editNotes}
                        onChange={(e) => setEditNotes(e.target.value)}
                        maxLength={150}
                        rows={2}
                      />
                    </label>
                  </div>
                ) : (
                  <div className="stack">
                    <div>
                      <strong>Title:</strong> {sessionDetails.title}
                    </div>

                    <div>
                      <strong>YouTube:</strong>{' '}
                      {sessionDetails.youtube_url ? (
                        <a
                          href={sessionDetails.youtube_url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          link
                        </a>
                      ) : (
                        <span>—</span>
                      )}
                    </div>

                    {sessionDetails.notes && (
                      <div className="session-notes-full">
                        <strong>Notes:</strong>{' '}
                        <span className="notes-text">{sessionDetails.notes}</span>
                      </div>
                    )}

                    <div title={sessionDetails.media_path}>
                      <strong>Media:</strong>{' '}
                      {sessionDetails.media_path
                        ? basename(sessionDetails.media_path)
                        : '—'}
                    </div>

                    {sessionDetails.processing_duration_seconds != null && (
                      <div>
                        <strong>Processed in:</strong>{' '}
                        {formatDuration(sessionDetails.processing_duration_seconds)}
                      </div>
                    )}
                  </div>
                )}
              </section>
            )}
          </div>
          </div>
          )}

          {sidebarTab === 'explore' && (
          <div className="search-tab" ref={chunkListRef}>
            <div className="search-controls">
              <input
                type="text"
                className="search-input"
                placeholder={'Search: reset / baron / "we should"'}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && canSearch) {
                    handleSearch()
                  }
                }}
                disabled={!canSearch}
              />
              <div className="search-filters-row">
                <div className="search-filters">
                  <input
                    type="text"
                    className="time-input"
                    placeholder="Start"
                    value={startTime}
                    onChange={(e) => {
                      setStartTime(e.target.value)
                      setTimeRangeError('')
                    }}
                    disabled={!canSearch}
                  />
                  <span className="time-separator">to</span>
                  <input
                    type="text"
                    className="time-input"
                    placeholder="End"
                    value={endTime}
                    onChange={(e) => {
                      setEndTime(e.target.value)
                      setTimeRangeError('')
                    }}
                    disabled={!canSearch}
                  />
                  <button
                    type="button"
                    className="action-btn"
                    onClick={handleSearch}
                    disabled={!canSearch}
                  >
                    Go
                  </button>
                  <button
                    type="button"
                    className="action-btn ghost"
                    onClick={handleClearSearch}
                    disabled={!canSearch}
                  >
                    Clear
                  </button>
                </div>
                {sessionId && (
                  <span className="search-status">
                    {isSearching ? (
                      <>
                        Results: {chunks.length}
                        {lastQuery && ` (${lastQuery})`}
                        {lastTimeRange && ` (${lastTimeRange})`}
                      </>
                    ) : (
                      `${chunks.length} chunks`
                    )}
                  </span>
                )}
              </div>
              {timeRangeError && (
                <span className="hint danger">{timeRangeError}</span>
              )}
              {!canSearch && (
                <span className="hint">Select a session to search</span>
              )}
            </div>

            {!sessionId && (
              <p className="hint">Select a session to view chunks.</p>
            )}
            {sessionId && chunks.length === 0 && (
              <p className="hint">No chunks yet. Process the uploaded file.</p>
            )}
            {sessionId && chunks.length > 0 && (
              <div className="pagination">
                <div className="actions">
                  {sessionDetails?.youtube_url && (
                    <label className="toggle-label-inline">
                      <input
                        type="checkbox"
                        checked={autoScrollEnabled}
                        onChange={(e) => setAutoScrollEnabled(e.target.checked)}
                      />
                      <span>Auto-scroll</span>
                    </label>
                  )}
                  <button
                    type="button"
                    className="action-btn ghost"
                    onClick={() => setPageIndex((prev) => Math.max(0, prev - 1))}
                    disabled={pageIndex === 0}
                  >
                    Prev
                  </button>
                  <button
                    type="button"
                    className="action-btn ghost"
                    onClick={() =>
                      setPageIndex((prev) => Math.min(totalPages - 1, prev + 1))
                    }
                    disabled={pageIndex >= totalPages - 1}
                  >
                    Next
                  </button>
                </div>
                <span className="hint">
                  Page {pageIndex + 1} / {totalPages}
                </span>
              </div>
            )}
            <div className="chunk-list">
              {pageChunks.map((chunk) => {
                const chunkKey =
                  chunk.id ??
                  `${chunk.start_ms}-${chunk.end_ms}-${chunk.text?.length ?? 0}`
                const isExpanded = expandedChunkIds.has(chunkKey)
                const previewText = getPreviewText(chunk.text || '')
                const youtubeLink = buildYoutubeUrlWithTimestamp(
                  sessionDetails?.youtube_url,
                  chunk.start_ms
                )
                return (
                  <div
                    key={chunkKey}
                    data-chunk-key={chunkKey}
                    className={`chunk ${chunkKey === activeChunkId ? 'active' : ''}`}
                  >
                    <div className="chunk-header">
                      {sessionDetails?.youtube_url ? (
                        <button
                          type="button"
                          className="chunk-times chunk-times-link"
                          onClick={() => handleTimestampClick(chunk)}
                        >
                          <span>{formatTime(chunk.start_ms)}</span>
                          <span>→</span>
                          <span>{formatTime(chunk.end_ms)}</span>
                        </button>
                      ) : (
                        <div className="chunk-times">
                          <span>{formatTime(chunk.start_ms)}</span>
                          <span>→</span>
                          <span>{formatTime(chunk.end_ms)}</span>
                        </div>
                      )}
                      {youtubeLink && (
                        <button
                          type="button"
                          className="copy-btn"
                          onClick={() => handleCopyTimestamp(chunkKey, youtubeLink)}
                          title="Copy timestamp URL"
                        >
                          {copiedChunkId === chunkKey ? '✓ Copied' : 'Copy'}
                        </button>
                      )}
                    </div>
                    <p
                      className="chunk-text"
                      onClick={() => toggleChunkExpanded(chunkKey)}
                      title={isExpanded ? 'Click to collapse' : 'Click to expand'}
                    >
                      {isExpanded ? chunk.text : previewText}
                    </p>
                  </div>
                )
              })}
            </div>
          </div>
          )}
          </div>
        </div>

        <div className="main-content">
          {sessionDetails?.youtube_url ? (
            <div className="youtube-player-container">
              <div id="youtube-player"></div>
            </div>
          ) : (
            <div className="no-player-message">
              <p className="hint">
                {sessionId
                  ? 'This session has no YouTube URL. Add one in the Sessions tab to see the embedded player.'
                  : 'Select a session with a YouTube URL to view the player.'}
              </p>
            </div>
          )}
        </div>
      </section>
    </div>
  )
}

export default App
