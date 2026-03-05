import { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import styles from './TranscriptPanel.module.css'
import { formatTime, parseScorecard, averageScore, getSpeakerDisplayName } from '../utils/formatters'
import { SPEAKER_COLORS, getSpeakerColor, TAG_COLORS, TAG_LABELS } from './SessionTimeline'

const TAG_PATTERN = /^\[(\w+)\]\s*(.*)$/

const getScoreColor = (score) => {
  if (score >= 7) return '#86EFAC'
  if (score >= 4) return '#FDE047'
  return '#FCA5A5'
}

export const TranscriptPanel = ({
  chunks,
  sessionDetails,
  activeChunkId,
  autoScrollEnabled,
  chunkListRef,
  onTimestampClick,
  onUpdateChunk,
  onCloseSession,
  onOpenSpeakerModal,
}) => {
  const [filterQuery, setFilterQuery] = useState('')
  const [editingChunkId, setEditingChunkId] = useState(null)
  const [editText, setEditText] = useState('')
  const [scorecardOpen, setScorecardOpen] = useState(false)
  const [editingNoteChunkId, setEditingNoteChunkId] = useState(null)
  const [noteText, setNoteText] = useState('')

  const transcriptRef = useRef(null)

  // Parse scorecard from session notes
  const scorecard = useMemo(
    () => parseScorecard(sessionDetails?.notes),
    [sessionDetails?.notes]
  )
  const avgScore = useMemo(() => averageScore(scorecard), [scorecard])

  // Client-side filtering
  const filteredChunks = useMemo(() => {
    if (!filterQuery.trim()) return chunks
    const q = filterQuery.toLowerCase()
    const names = sessionDetails?.speaker_names
    return chunks.filter(
      (chunk) =>
        (chunk.text || '').toLowerCase().includes(q) ||
        (chunk.speaker || '').toLowerCase().includes(q) ||
        (getSpeakerDisplayName(chunk.speaker, names) || '').toLowerCase().includes(q) ||
        (chunk.notes || '').toLowerCase().includes(q)
    )
  }, [chunks, filterQuery, sessionDetails?.speaker_names])

  // Extract LLM tag from chunk notes
  const getChunkTag = useCallback((chunk) => {
    if (!chunk.notes) return null
    const lines = chunk.notes.split('\n')
    for (const line of lines) {
      const match = line.trim().match(TAG_PATTERN)
      if (match && TAG_COLORS[match[1]]) {
        return { type: match[1], color: TAG_COLORS[match[1]] }
      }
    }
    return null
  }, [])

  // Double-click editing
  const handleDoubleClick = useCallback((chunk, chunkKey) => {
    setEditingChunkId(chunkKey)
    setEditText(chunk.text || '')
  }, [])

  const handleSaveEdit = useCallback(async (chunk) => {
    const trimmed = editText.trim()
    if (!trimmed || !chunk.id) {
      setEditingChunkId(null)
      setEditText('')
      return
    }
    try {
      await onUpdateChunk(chunk.id, { text: trimmed })
    } catch {
      // Error handled by parent
    }
    setEditingChunkId(null)
    setEditText('')
  }, [editText, onUpdateChunk])

  const handleCancelEdit = useCallback(() => {
    setEditingChunkId(null)
    setEditText('')
  }, [])

  // Note editing
  const handleEditNote = useCallback((chunk, chunkKey) => {
    setEditingNoteChunkId(chunkKey)
    setNoteText(chunk.notes || '')
  }, [])

  const handleSaveNote = useCallback(async (chunk) => {
    if (!chunk.id) return
    try {
      await onUpdateChunk(chunk.id, { notes: noteText.trim() || null })
    } catch {
      // Error handled by parent
    }
    setEditingNoteChunkId(null)
    setNoteText('')
  }, [noteText, onUpdateChunk])

  const handleCancelNote = useCallback(() => {
    setEditingNoteChunkId(null)
    setNoteText('')
  }, [])

  // Bookmark toggle
  const handleToggleBookmark = useCallback(async (chunk) => {
    if (!chunk.id) return
    try {
      await onUpdateChunk(chunk.id, { is_bookmarked: chunk.is_bookmarked ? 0 : 1 })
    } catch {
      // Error handled by parent
    }
  }, [onUpdateChunk])

  // Auto-scroll to active chunk
  useEffect(() => {
    if (!activeChunkId || !autoScrollEnabled || !transcriptRef.current) return
    const el = transcriptRef.current.querySelector(
      `[data-chunk-key="${activeChunkId}"]`
    )
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [activeChunkId, autoScrollEnabled])

  // Wire chunkListRef to transcript body for useYouTubePlayer scroll detection
  const setTranscriptRef = useCallback((el) => {
    transcriptRef.current = el
    if (chunkListRef) chunkListRef.current = el
  }, [chunkListRef])

  return (
    <div className={styles.panel}>
      {/* Top bar */}
      <div className={styles.topBar}>
        <button className={styles.backBtn} onClick={onCloseSession} title="Back to sessions">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M9 2L4 7L9 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
        <h2 className={styles.title}>
          {sessionDetails?.title || 'Untitled session'}
        </h2>
        {onOpenSpeakerModal && (
          <button
            className={styles.assignSpeakersBtn}
            onClick={onOpenSpeakerModal}
            title="Assign speaker names"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <circle cx="5" cy="4.5" r="2" stroke="currentColor" strokeWidth="1.2"/>
              <path d="M1 11.5C1 9.29 2.79 7.5 5 7.5C7.21 7.5 9 9.29 9 11.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
              <path d="M10 6.5L10 10.5M8 8.5L12 8.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
            </svg>
          </button>
        )}
      </div>

      {/* Scorecard (collapsible) */}
      {scorecard && (
        <div className={styles.scorecard}>
          <button
            className={styles.scorecardToggle}
            onClick={() => setScorecardOpen(!scorecardOpen)}
          >
            <span className={styles.scorecardLabel}>Scorecard</span>
            <span className={styles.avgScore}>{avgScore}</span>
            <span className={styles.toggleArrow}>{scorecardOpen ? '▲' : '▼'}</span>
          </button>
          {scorecardOpen && (
            <div className={styles.scorecardBody}>
              {Object.entries(scorecard).map(([category, score]) => {
                const color = getScoreColor(score)
                return (
                  <div key={category} className={styles.scorecardRow}>
                    <div className={styles.scorecardRowTop}>
                      <span className={styles.scorecardCategory}>{category}</span>
                      <span
                        className={styles.scorecardScore}
                        style={{ color }}
                      >
                        {score}/10
                      </span>
                    </div>
                    <div className={styles.scoreBar}>
                      <div
                        className={styles.scoreBarFill}
                        style={{
                          width: `${(score / 10) * 100}%`,
                          background: color,
                          opacity: 0.5,
                        }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {/* Filter bar */}
      <div className={styles.filterBar}>
        <input
          type="text"
          className={styles.filterInput}
          placeholder="Filter transcript..."
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
        />
        {filterQuery && (
          <>
            <span className={styles.filterCount}>
              {filteredChunks.length}/{chunks.length}
            </span>
            <button
              className={styles.clearBtn}
              onClick={() => setFilterQuery('')}
              title="Clear filter"
            >
              ✕
            </button>
          </>
        )}
      </div>

      {/* Transcript body */}
      <div className={styles.transcriptBody} ref={setTranscriptRef}>
        {filteredChunks.length === 0 && (
          <div className={styles.emptyState}>
            {filterQuery ? 'No matching chunks' : 'No chunks'}
          </div>
        )}
        {filteredChunks.map((chunk) => {
          const chunkKey =
            chunk.id ?? `${chunk.start_ms}-${chunk.end_ms}-${chunk.text?.length ?? 0}`
          const tag = getChunkTag(chunk)
          const speakerIdx = parseInt(chunk.speaker?.match(/(\d+)/)?.[1] ?? '0', 10)
          const speakerColor = getSpeakerColor(speakerIdx)
          const displayName = getSpeakerDisplayName(chunk.speaker, sessionDetails?.speaker_names)
          const isActive = chunkKey === activeChunkId
          const isEditing = editingChunkId === chunkKey
          const isEditingNote = editingNoteChunkId === chunkKey
          const hasNote = chunk.notes && !TAG_PATTERN.test(chunk.notes.split('\n')[0]?.trim())

          return (
            <div
              key={chunkKey}
              data-chunk-key={chunkKey}
              className={`${styles.chunkRow} ${isActive ? styles.active : ''}`}
            >
              {/* Tag dot */}
              {tag && (
                <span
                  className={styles.tagDot}
                  style={{ background: tag.color }}
                  title={TAG_LABELS[tag.type] || tag.type}
                />
              )}

              {/* Timestamp */}
              <button
                className={styles.timestamp}
                onClick={() => onTimestampClick(chunk)}
              >
                {formatTime(chunk.start_ms)}
              </button>

              {/* Speaker label */}
              {displayName && (
                <span
                  className={styles.speaker}
                  style={{ color: speakerColor }}
                >
                  {displayName}:
                </span>
              )}

              {/* Text */}
              {isEditing ? (
                <textarea
                  className={styles.editTextarea}
                  value={editText}
                  onChange={(e) => setEditText(e.target.value)}
                  onBlur={() => handleSaveEdit(chunk)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      handleSaveEdit(chunk)
                    } else if (e.key === 'Escape') {
                      handleCancelEdit()
                    }
                  }}
                  maxLength={1000}
                  autoFocus
                />
              ) : (
                <span
                  className={styles.chunkText}
                  onDoubleClick={() => handleDoubleClick(chunk, chunkKey)}
                  title="Double-click to edit"
                >
                  {chunk.text}
                </span>
              )}

              {/* Inline action icons */}
              <span className={styles.chunkActions}>
                <button
                  className={`${styles.chunkActionBtn} ${chunk.is_bookmarked ? styles.alwaysVisible : styles.hoverVisible}`}
                  onClick={() => handleToggleBookmark(chunk)}
                  title={chunk.is_bookmarked ? 'Remove bookmark' : 'Bookmark'}
                >
                  {chunk.is_bookmarked ? '⭐' : '☆'}
                </button>
                <button
                  className={`${styles.chunkActionBtn} ${hasNote || chunk.notes ? styles.alwaysVisible : styles.hoverVisible}`}
                  onClick={() => handleEditNote(chunk, chunkKey)}
                  title={chunk.notes ? 'Edit note' : 'Add note'}
                >
                  {chunk.notes ? '📝' : '✏️'}
                </button>
              </span>

              {/* Note editing row */}
              {isEditingNote && (
                <div className={styles.noteRow}>
                  <textarea
                    className={styles.noteTextarea}
                    value={noteText}
                    onChange={(e) => setNoteText(e.target.value)}
                    onBlur={() => handleSaveNote(chunk)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault()
                        handleSaveNote(chunk)
                      } else if (e.key === 'Escape') {
                        handleCancelNote()
                      }
                    }}
                    maxLength={100}
                    rows={2}
                    placeholder="Add note..."
                    autoFocus
                  />
                </div>
              )}

              {/* Note display */}
              {!isEditingNote && chunk.notes && hasNote && (
                <div
                  className={styles.noteDisplay}
                  onClick={() => handleEditNote(chunk, chunkKey)}
                  title="Click to edit note"
                >
                  {chunk.notes}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
