import { useState, useEffect, useMemo } from 'react'
import styles from './ExplorePanel.module.css'
import * as formatters from '../utils/formatters'
import { PAGE_SIZE } from '../utils/api'
import { TAG_COLORS, TAG_LABELS, getSpeakerColor } from './SessionTimeline'

const TAG_PATTERN = /^\[(\w+)\]\s*(.*)$/

const extractTags = (notes) => {
  if (!notes) return []
  const tags = []
  for (const line of notes.split('\n')) {
    const match = line.trim().match(TAG_PATTERN)
    if (match && TAG_COLORS[match[1]]) {
      tags.push({ type: match[1], label: match[2] })
    }
  }
  return tags
}

const highlightText = (text, query) => {
  if (!query || !text) return text
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const regex = new RegExp(`(${escaped})`, 'gi')
  const parts = text.split(regex)
  if (parts.length === 1) return text
  return parts.map((part, i) =>
    regex.test(part)
      ? <mark key={i} className={styles.highlight}>{part}</mark>
      : part
  )
}

/**
 * Explore view — full-width search across all sessions
 */
export const ExplorePanel = ({
  sessions,
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
  onNavigateToSession,
}) => {
  // Search state
  const [searchQuery, setSearchQuery] = useState('')
  const [startTime, setStartTime] = useState('')
  const [endTime, setEndTime] = useState('')
  const [bookmarkedOnly, setBookmarkedOnly] = useState(false)
  const [hasNotesOnly, setHasNotesOnly] = useState(false)
  const [hasTagsOnly, setHasTagsOnly] = useState(false)
  const [timeRangeError, setTimeRangeError] = useState('')
  const [isSearching, setIsSearching] = useState(false)
  const [lastQuery, setLastQuery] = useState('')
  const [lastTimeRange, setLastTimeRange] = useState('')
  const [filterSessionId, setFilterSessionId] = useState('')

  // Chunk interaction state
  const [expandedChunkIds, setExpandedChunkIds] = useState(new Set())
  const [copiedChunkId, setCopiedChunkId] = useState(null)
  const [editingNoteChunkId, setEditingNoteChunkId] = useState(null)
  const [noteText, setNoteText] = useState('')
  const [editingTextChunkId, setEditingTextChunkId] = useState(null)
  const [chunkText, setChunkText] = useState('')

  // Re-run search when session filter changes while results are showing
  const handleFilterSessionChange = async (newFilterId) => {
    setFilterSessionId(newFilterId)
    if (isSearching) {
      // Re-search with the new session filter using current search params
      const trimmed = searchQuery.trim()
      const startMs = startTime.trim() ? formatters.parseTime(startTime.trim()) : null
      const endMs = endTime.trim() ? formatters.parseTime(endTime.trim()) : null
      try {
        await onSearch(newFilterId || null, {
          query: trimmed,
          limit: 50,
          start_time_ms: startMs,
          end_time_ms: endMs,
          is_bookmarked: bookmarkedOnly || null,
        })
        setPageIndex(0)
        setExpandedChunkIds(new Set())
      } catch (err) {
        // Error handled by parent
      }
    }
  }

  // Session lookup map for O(1) metadata access
  const sessionMap = useMemo(() => {
    const map = {}
    if (sessions) {
      for (const s of sessions) {
        map[s.id] = s
      }
    }
    return map
  }, [sessions])

  // Reset page index when chunks change
  useEffect(() => {
    const totalPages = Math.max(1, Math.ceil(chunks.length / PAGE_SIZE))
    if (pageIndex > totalPages - 1) {
      setPageIndex(0)
    }
  }, [chunks.length, pageIndex])

  const totalPages = Math.max(1, Math.ceil(chunks.length / PAGE_SIZE))
  const pageStart = pageIndex * PAGE_SIZE
  const pageChunks = chunks.slice(pageStart, pageStart + PAGE_SIZE)

  // Group page chunks by session when doing multi-session search
  const groupedResults = useMemo(() => {
    if (!isSearching || filterSessionId || pageChunks.length === 0) return null
    // Check if results span multiple sessions
    const sessionIds = new Set(pageChunks.map(c => c.session_id))
    if (sessionIds.size <= 1) return null

    const groups = []
    const seen = new Map()
    for (const chunk of pageChunks) {
      const sid = chunk.session_id
      if (!seen.has(sid)) {
        seen.set(sid, groups.length)
        groups.push({ sessionId: sid, session: sessionMap[sid], chunks: [] })
      }
      groups[seen.get(sid)].chunks.push(chunk)
    }
    return groups
  }, [pageChunks, isSearching, filterSessionId, sessionMap])

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
      if (filterSessionId) {
        await onReloadSession(filterSessionId)
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

    setTimeRangeError('')

    try {
      await onSearch(filterSessionId || null, {
        query: trimmed,
        limit: 50,
        start_time_ms: startMs,
        end_time_ms: endMs,
        is_bookmarked: bookmarkedOnly || null,
      })
      setIsSearching(true)
      setLastQuery(trimmed)

      if (startTimeInput || endTimeInput) {
        setLastTimeRange(`${startTimeInput || '0:00'}-${endTimeInput || '∞'}`)
      } else {
        setLastTimeRange('')
      }

      setPageIndex(0)
      setExpandedChunkIds(new Set())
    } catch (err) {
      // Error handled by parent
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

    if (filterSessionId) {
      await onReloadSession(filterSessionId)
    } else {
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
    // Cross-session navigation
    if (chunk.session_id && chunk.session_id !== sessionId && onNavigateToSession) {
      onNavigateToSession(chunk.session_id, chunk)
      return
    }
    // Same session or no session context — just seek
    if (onTimestampClick) {
      onTimestampClick(chunk)
    } else {
      const url = chunk.youtube_url || sessionDetails?.youtube_url
      const youtubeLink = formatters.buildYoutubeUrlWithTimestamp(url, chunk.start_ms)
      if (youtubeLink) window.open(youtubeLink, '_blank')
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
      // Error handled by parent
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
    if (!trimmedText) return
    try {
      await onUpdateChunk(chunk.id, { text: trimmedText })
      setEditingTextChunkId(null)
      setChunkText('')
    } catch (err) {
      // Error handled by parent
    }
  }

  const handleToggleBookmark = async (chunk) => {
    if (!chunk.id) return
    const newBookmarkState = chunk.is_bookmarked ? 0 : 1
    try {
      await onUpdateChunk(chunk.id, { is_bookmarked: newBookmarkState })
    } catch (err) {
      // Error handled by parent
    }
  }

  const renderChunk = (chunk) => {
    const chunkKey =
      chunk.id ?? `${chunk.start_ms}-${chunk.end_ms}-${chunk.text?.length ?? 0}`
    const isExpanded = expandedChunkIds.has(chunkKey)
    const previewText = formatters.getPreviewText(chunk.text || '')
    const chunkSession = sessionMap[chunk.session_id] || sessionDetails
    const chunkVideoUrl = chunk.youtube_url || chunkSession?.youtube_url
    const youtubeLink = formatters.buildYoutubeUrlWithTimestamp(
      chunkVideoUrl,
      chunk.start_ms
    )
    const speakerName = formatters.getSpeakerDisplayName(
      chunk.speaker,
      chunkSession?.speaker_names
    )
    const speakerIndex = parseInt(chunk.speaker?.match(/(\d+)/)?.[1] ?? '0', 10)
    const tags = extractTags(chunk.notes)

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
              <span
                className={styles.speakerDot}
                style={{ background: getSpeakerColor(speakerIndex) }}
              />
              {speakerName}
            </span>
          )}
          {tags.length > 0 && (
            <span className={styles.tagDots}>
              {tags.map((tag, i) => (
                <span
                  key={i}
                  className={styles.tagDot}
                  style={{ background: TAG_COLORS[tag.type] }}
                  title={`${TAG_LABELS[tag.type]}: ${tag.label}`}
                />
              ))}
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
            {isExpanded
              ? highlightText(chunk.text, lastQuery)
              : highlightText(previewText, lastQuery)}
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
  }

  return (
    <div className={styles.exploreContainer} ref={chunkListRef}>
      <div className={styles.exploreContent}>
        {/* Header */}
        <div className={styles.exploreHeader}>
          <h1 className={styles.exploreTitle}>Explore</h1>
        </div>

        {/* Search Bar */}
        <div className={styles.searchBar}>
          <div className={styles.searchInputWrapper}>
            <svg className={styles.searchIcon} viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM2 9a7 7 0 1112.452 4.391l3.328 3.329a.75.75 0 11-1.06 1.06l-3.329-3.328A7 7 0 012 9z" clipRule="evenodd" />
            </svg>
            <input
              type="text"
              className={styles.searchInput}
              placeholder="Search across all sessions..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSearch()
              }}
            />
          </div>
        </div>

        {/* Filters */}
        <div className={styles.filtersRow}>
          <select
            className={styles.sessionFilter}
            value={filterSessionId}
            onChange={(e) => handleFilterSessionChange(e.target.value)}
          >
            <option value="">All sessions</option>
            {(sessions || [])
              .filter(s => s.status === 'ready')
              .map(s => (
                <option key={s.id} value={s.id}>
                  {s.title || 'Untitled'}
                </option>
              ))}
          </select>
          <div className={styles.timeFilters}>
            <input
              type="text"
              className={styles.timeInput}
              placeholder="Start"
              value={startTime}
              onChange={(e) => {
                setStartTime(e.target.value)
                setTimeRangeError('')
              }}
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
            />
          </div>
          <label className={styles.bookmarkFilterLabel}>
            <input
              type="checkbox"
              checked={bookmarkedOnly}
              onChange={(e) => setBookmarkedOnly(e.target.checked)}
            />
            <span>Bookmarked</span>
          </label>
          <div className={styles.filterActions}>
            <button
              type="button"
              className={`${styles.actionBtn} ${styles.primary}`}
              onClick={handleSearch}
            >
              Search
            </button>
            {isSearching && (
              <button
                type="button"
                className={styles.actionBtn}
                onClick={handleClearSearch}
              >
                Clear
              </button>
            )}
          </div>
        </div>

        {timeRangeError && (
          <span className="hint danger">{timeRangeError}</span>
        )}

        {/* Empty state: before searching */}
        {!isSearching && chunks.length === 0 && (
          <div className={styles.emptyState}>
            <svg className={styles.emptyIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
            </svg>
            <p className={styles.emptyTitle}>Search your scrim comms</p>
            <p className={styles.emptySubtext}>
              Find any callout, decision, or moment across all sessions
            </p>
          </div>
        )}

        {/* Empty state: no results */}
        {isSearching && chunks.length === 0 && (
          <div className={styles.emptyState}>
            <p className={styles.emptyTitle}>No results found</p>
            <p className={styles.emptySubtext}>
              {lastQuery
                ? <>No results found for &ldquo;{lastQuery}&rdquo;</>
                : 'Try different search terms or filters'}
            </p>
          </div>
        )}

        {/* Results header + pagination */}
        {chunks.length > 0 && (
          <div className={styles.resultsHeader}>
            <span className={styles.resultCount}>
              {chunks.length} result{chunks.length !== 1 ? 's' : ''}
              {lastQuery && <> for &ldquo;{lastQuery}&rdquo;</>}
              {lastTimeRange && <> ({lastTimeRange})</>}
            </span>
            <div className={styles.paginationControls}>
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
                className={styles.paginationBtn}
                onClick={() => setPageIndex((prev) => Math.max(0, prev - 1))}
                disabled={pageIndex === 0}
              >
                Prev
              </button>
              <span>
                {pageIndex + 1} / {totalPages}
              </span>
              <button
                type="button"
                className={styles.paginationBtn}
                onClick={() =>
                  setPageIndex((prev) => Math.min(totalPages - 1, prev + 1))
                }
                disabled={pageIndex >= totalPages - 1}
              >
                Next
              </button>
            </div>
          </div>
        )}

        {/* Results list */}
        <div className={styles.resultsList}>
          {groupedResults ? (
            groupedResults.map((group) => (
              <div key={group.sessionId} className={styles.sessionGroup}>
                <div className={styles.sessionGroupHeader}>
                  <span className={styles.sessionGroupTitle}>
                    {group.session?.title || 'Untitled Session'}
                  </span>
                  <span className={styles.sessionGroupDate}>
                    {group.session?.created_at
                      ? formatters.formatRelativeTime(group.session.created_at)
                      : ''}
                  </span>
                </div>
                {group.chunks.map((chunk) => renderChunk(chunk))}
              </div>
            ))
          ) : (
            pageChunks.map((chunk) => renderChunk(chunk))
          )}
        </div>
      </div>
    </div>
  )
}
