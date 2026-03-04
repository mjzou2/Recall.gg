import { useState, useEffect } from 'react'
import styles from './ExplorePanel.module.css'
import * as formatters from '../utils/formatters'
import { PAGE_SIZE } from '../utils/api'

/**
 * Complete explore/search tab with filters, chunk list, and pagination
 *
 * @param {Object} props
 * @param {string} props.sessionId - ID of currently active session
 * @param {Object} props.sessionDetails - Details of active session
 * @param {Array} props.chunks - Array of chunk objects
 * @param {Object} props.youtubePlayer - YouTube player instance
 * @param {Object} props.chunkListRef - Ref to chunk list container
 * @param {string} props.activeChunkId - ID of currently active chunk
 * @param {boolean} props.autoScrollEnabled - Whether auto-scroll is enabled
 * @param {Function} props.setAutoScrollEnabled - Handler to toggle auto-scroll
 * @param {number} props.pageIndex - Current page index
 * @param {Function} props.setPageIndex - Handler to update page index
 * @param {Function} props.onSearch - Handler to search chunks
 * @param {Function} props.onUpdateChunk - Handler to update chunk
 * @param {Function} props.onTimestampClick - Handler for timestamp clicks
 */
export const ExplorePanel = ({
  sessionId,
  sessionDetails,
  chunks,
  youtubePlayer,
  chunkListRef,
  activeChunkId,
  autoScrollEnabled,
  setAutoScrollEnabled,
  pageIndex,
  setPageIndex,
  onSearch,
  onReloadSession,
  onUpdateChunk,
  onTimestampClick,
}) => {
  // Search state
  const [searchQuery, setSearchQuery] = useState('')
  const [startTime, setStartTime] = useState('')
  const [endTime, setEndTime] = useState('')
  const [bookmarkedOnly, setBookmarkedOnly] = useState(false)
  const [timeRangeError, setTimeRangeError] = useState('')
  const [isSearching, setIsSearching] = useState(false)
  const [lastQuery, setLastQuery] = useState('')
  const [lastTimeRange, setLastTimeRange] = useState('')

  // Chunk interaction state
  const [expandedChunkIds, setExpandedChunkIds] = useState(new Set())
  const [copiedChunkId, setCopiedChunkId] = useState(null)
  const [editingNoteChunkId, setEditingNoteChunkId] = useState(null)
  const [noteText, setNoteText] = useState('')
  const [editingTextChunkId, setEditingTextChunkId] = useState(null)
  const [chunkText, setChunkText] = useState('')

  const canSearch = true

  // Reset page index when chunks change
  useEffect(() => {
    const totalPages = Math.max(1, Math.ceil(chunks.length / PAGE_SIZE))
    if (pageIndex > totalPages - 1) {
      setPageIndex(0)
    }
  }, [chunks.length, pageIndex])

  const handleSearch = async () => {
    const trimmed = searchQuery.trim()
    const startTimeInput = startTime.trim()
    const endTimeInput = endTime.trim()

    // If no filters provided, reset
    if (!trimmed && !startTimeInput && !endTimeInput && !bookmarkedOnly) {
      setSearchQuery('')
      setStartTime('')
      setEndTime('')
      setBookmarkedOnly(false)
      setIsSearching(false)
      setLastQuery('')
      setLastTimeRange('')
      setTimeRangeError('')
      setPageIndex(0)
      setExpandedChunkIds(new Set())
      // Reload session chunks if a session is selected
      if (sessionId) {
        await onReloadSession(sessionId)
      }
      return
    }

    // Parse time inputs
    const startMs = startTimeInput ? formatters.parseTime(startTimeInput) : null
    const endMs = endTimeInput ? formatters.parseTime(endTimeInput) : null

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

    try {
      await onSearch(sessionId || null, {
        query: trimmed,
        limit: 50,
        start_time_ms: startMs,
        end_time_ms: endMs,
        is_bookmarked: bookmarkedOnly || null,
      })
      setIsSearching(true)
      setLastQuery(trimmed)

      // Track time range for display
      if (startTimeInput || endTimeInput) {
        setLastTimeRange(`${startTimeInput || '0:00'}-${endTimeInput || '∞'}`)
      } else {
        setLastTimeRange('')
      }

      // Reset chunk views
      setPageIndex(0)
      setExpandedChunkIds(new Set())
    } catch (err) {
      // Error is handled by parent
    }
  }

  const handleClearSearch = async () => {
    setSearchQuery('')
    setStartTime('')
    setEndTime('')
    setBookmarkedOnly(false)
    setTimeRangeError('')
    setIsSearching(false)
    setLastQuery('')
    setLastTimeRange('')
    setPageIndex(0)
    setExpandedChunkIds(new Set())

    if (sessionId) {
      await onReloadSession(sessionId)
    } else {
      // No session — clear results via empty search
      await onSearch(null, { query: '' })
    }
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
      setTimeout(() => setCopiedChunkId(null), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  const handleTimestampClickInternal = (chunk) => {
    // Use the prop handler if available, otherwise fallback to opening in new tab
    if (onTimestampClick) {
      onTimestampClick(chunk)
    } else {
      const url = chunk.youtube_url || sessionDetails?.youtube_url
      const youtubeLink = formatters.buildYoutubeUrlWithTimestamp(url, chunk.start_ms)
      if (youtubeLink) {
        window.open(youtubeLink, '_blank')
      }
    }
  }

  const handleEditNote = (chunk, chunkKey) => {
    setEditingNoteChunkId(chunkKey)
    setNoteText(chunk.notes || '')
  }

  const handleCancelNote = () => {
    setEditingNoteChunkId(null)
    setNoteText('')
  }

  const handleSaveNote = async (chunk) => {
    if (!chunk.id) return
    try {
      await onUpdateChunk(chunk.id, { notes: noteText.trim() || null })
      setEditingNoteChunkId(null)
      setNoteText('')
    } catch (err) {
      // Error is handled by parent
    }
  }

  const handleEditText = (chunk, chunkKey) => {
    setEditingTextChunkId(chunkKey)
    setChunkText(chunk.text || '')
  }

  const handleCancelText = () => {
    setEditingTextChunkId(null)
    setChunkText('')
  }

  const handleSaveText = async (chunk) => {
    if (!chunk.id) return
    const trimmedText = chunkText.trim()
    if (!trimmedText) {
      return
    }
    try {
      await onUpdateChunk(chunk.id, { text: trimmedText })
      setEditingTextChunkId(null)
      setChunkText('')
    } catch (err) {
      // Error is handled by parent
    }
  }

  const handleToggleBookmark = async (chunk) => {
    if (!chunk.id) return
    const newBookmarkState = chunk.is_bookmarked ? 0 : 1
    try {
      await onUpdateChunk(chunk.id, { is_bookmarked: newBookmarkState })
    } catch (err) {
      // Error is handled by parent
    }
  }

  const totalPages = Math.max(1, Math.ceil(chunks.length / PAGE_SIZE))
  const pageStart = pageIndex * PAGE_SIZE
  const pageChunks = chunks.slice(pageStart, pageStart + PAGE_SIZE)

  return (
    <div className={styles.searchTab} ref={chunkListRef}>
      <div className={styles.searchControls}>
        <input
          type="text"
          className={styles.searchInput}
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
        <div className={styles.searchFiltersRow}>
          <div className={styles.searchFilters}>
            <input
              type="text"
              className={styles.timeInput}
              placeholder="Start"
              value={startTime}
              onChange={(e) => {
                setStartTime(e.target.value)
                setTimeRangeError('')
              }}
              disabled={!canSearch}
            />
            <span className={styles.timeSeparator}>to</span>
            <input
              type="text"
              className={styles.timeInput}
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
              className={`${styles.actionBtn} ${styles.primary}`}
              onClick={handleSearch}
              disabled={!canSearch}
            >
              Go
            </button>
            <button
              type="button"
              className={styles.actionBtn}
              onClick={handleClearSearch}
              disabled={!canSearch}
            >
              Clear
            </button>
          </div>
          <label className={styles.bookmarkFilterLabel}>
            <input
              type="checkbox"
              checked={bookmarkedOnly}
              onChange={(e) => setBookmarkedOnly(e.target.checked)}
            />
            <span>Show bookmarked only</span>
          </label>
          {(sessionId || isSearching) && (
            <span className={styles.searchStatus}>
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
      </div>

      {!sessionId && chunks.length === 0 && !isSearching && (
        <p className="hint">Search across all sessions, or select a session to view its chunks.</p>
      )}
      {chunks.length === 0 && isSearching && (
        <p className="hint">No chunks found. Try different search terms.</p>
      )}
      {sessionId && chunks.length === 0 && !isSearching && (
        <p className="hint">No chunks found. Check the filters or process the uploaded file.</p>
      )}
      {chunks.length > 0 && (
        <div className={styles.pagination}>
          <div className="actions">
            {sessionDetails?.youtube_url && (
              <label className={styles.toggleLabelInline}>
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
              className={styles.actionBtn}
              onClick={() => setPageIndex((prev) => Math.max(0, prev - 1))}
              disabled={pageIndex === 0}
            >
              Prev
            </button>
            <button
              type="button"
              className={styles.actionBtn}
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
      <div className={styles.chunkList}>
        {pageChunks.map((chunk) => {
          const chunkKey =
            chunk.id ??
            `${chunk.start_ms}-${chunk.end_ms}-${chunk.text?.length ?? 0}`
          const isExpanded = expandedChunkIds.has(chunkKey)
          const previewText = formatters.getPreviewText(chunk.text || '')
          const chunkVideoUrl = chunk.youtube_url || sessionDetails?.youtube_url
          const youtubeLink = formatters.buildYoutubeUrlWithTimestamp(
            chunkVideoUrl,
            chunk.start_ms
          )
          return (
            <div
              key={chunkKey}
              data-chunk-key={chunkKey}
              className={`${styles.chunk} ${chunkKey === activeChunkId ? styles.active : ''}`}
            >
              <div className={styles.chunkHeader}>
                {chunkVideoUrl ? (
                  <button
                    type="button"
                    className={`${styles.chunkTimes} ${styles.chunkTimesLink}`}
                    onClick={() => handleTimestampClickInternal(chunk)}
                  >
                    <span>{formatters.formatTime(chunk.start_ms)}</span>
                    <span>→</span>
                    <span>{formatters.formatTime(chunk.end_ms)}</span>
                  </button>
                ) : (
                  <div className={styles.chunkTimes}>
                    <span>{formatters.formatTime(chunk.start_ms)}</span>
                    <span>→</span>
                    <span>{formatters.formatTime(chunk.end_ms)}</span>
                  </div>
                )}
                {chunk.speaker && (
                  <span className={styles.speakerLabel}>
                    {formatters.getSpeakerDisplayName(chunk.speaker, sessionDetails?.speaker_names)}
                  </span>
                )}
                <div className={styles.chunkHeaderRight}>
                  {youtubeLink && (
                    <button
                      type="button"
                      className={`${styles.copyBtn} ${isExpanded ? styles.alwaysVisible : styles.hoverVisible}`}
                      onClick={() => handleCopyTimestamp(chunkKey, youtubeLink)}
                      title="Copy timestamp URL"
                    >
                      {copiedChunkId === chunkKey ? '✓' : '📋'}
                    </button>
                  )}
                  <button
                    type="button"
                    className={`${styles.noteBtn} ${isExpanded || chunk.notes ? styles.alwaysVisible : styles.hoverVisible}`}
                    onClick={() => {
                      if (!isExpanded) {
                        toggleChunkExpanded(chunkKey)
                      } else {
                        handleEditNote(chunk, chunkKey)
                      }
                    }}
                    title={
                      !isExpanded && chunk.notes
                        ? 'Has note - click to expand'
                        : !isExpanded
                        ? 'Expand to add note'
                        : chunk.notes
                        ? 'Edit note'
                        : 'Add note'
                    }
                  >
                    {chunk.notes ? '📝' : '✏️'}
                  </button>
                  <button
                    type="button"
                    className={`${styles.copyBtn} ${isExpanded || chunk.is_bookmarked ? styles.alwaysVisible : styles.hoverVisible}`}
                    onClick={(e) => {
                      e.stopPropagation()
                      handleToggleBookmark(chunk)
                    }}
                    title={chunk.is_bookmarked ? 'Remove bookmark' : 'Add bookmark'}
                  >
                    {chunk.is_bookmarked ? '⭐' : '☆'}
                  </button>
                  <button
                    type="button"
                    className={isExpanded ? `${styles.collapseBtn} ${styles.alwaysVisible}` : `${styles.expandIndicator} ${styles.hoverVisible}`}
                    onClick={() => toggleChunkExpanded(chunkKey)}
                    title={isExpanded ? 'Click to collapse' : 'Click to expand'}
                  >
                    {isExpanded ? '▲' : '▼'}
                  </button>
                </div>
              </div>
              {editingTextChunkId === chunkKey ? (
                <div className={styles.chunkTextEdit}>
                  <label className="field">
                    <span>Text ({1000 - chunkText.length} chars left)</span>
                    <textarea
                      value={chunkText}
                      onChange={(e) => setChunkText(e.target.value)}
                      onBlur={() => handleSaveText(chunk)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault()
                          handleSaveText(chunk)
                        } else if (e.key === 'Escape') {
                          handleCancelText()
                        }
                      }}
                      maxLength={1000}
                      rows={5}
                      placeholder="Chunk text..."
                      autoFocus
                    />
                  </label>
                </div>
              ) : (
                <p
                  className={`${styles.chunkText} ${isExpanded && editingNoteChunkId !== chunkKey ? styles.editable : ''}`}
                  onClick={() => {
                    if (isExpanded && editingNoteChunkId !== chunkKey) {
                      handleEditText(chunk, chunkKey)
                    } else if (!isExpanded) {
                      toggleChunkExpanded(chunkKey)
                    }
                  }}
                  title={
                    isExpanded && editingNoteChunkId !== chunkKey
                      ? 'Click to edit'
                      : !isExpanded
                      ? 'Click to expand'
                      : ''
                  }
                >
                  {isExpanded ? chunk.text : previewText}
                </p>
              )}
              {isExpanded && editingNoteChunkId === chunkKey && (
                <div className={styles.chunkNoteEdit}>
                  <label className="field">
                    <span>Notes ({100 - noteText.length} chars left)</span>
                    <textarea
                      value={noteText}
                      onChange={(e) => setNoteText(e.target.value)}
                      onBlur={() => handleSaveNote(chunk)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          e.preventDefault()
                          handleSaveNote(chunk)
                        } else if (e.key === 'Escape') {
                          handleCancelNote()
                        }
                      }}
                      maxLength={100}
                      rows={3}
                      placeholder="Add your notes here..."
                      autoFocus
                    />
                  </label>
                </div>
              )}
              {isExpanded && editingNoteChunkId !== chunkKey && chunk.notes && (
                <div className={styles.chunkNoteDisplay}>
                  <p className={styles.noteLabel}>Notes:</p>
                  <p
                    className={styles.noteTextDisplay}
                    onClick={() => handleEditNote(chunk, chunkKey)}
                    title="Click to edit"
                  >
                    {chunk.notes}
                  </p>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
