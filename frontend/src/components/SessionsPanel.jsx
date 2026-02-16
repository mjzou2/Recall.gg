import { useState } from 'react'
import styles from './SessionsPanel.module.css'
import * as formatters from '../utils/formatters'

/**
 * Complete sessions tab with list, create form, and active session panel
 *
 * @param {Object} props
 * @param {Array} props.sessions - Array of all sessions
 * @param {string} props.sessionId - ID of currently active session
 * @param {Object} props.sessionDetails - Details of active session
 * @param {boolean} props.isUploading - Whether media is currently uploading
 * @param {boolean} props.isProcessing - Whether session is being processed
 * @param {number} props.elapsedSeconds - Elapsed processing time in seconds
 * @param {Function} props.onLoadSessions - Handler to reload sessions list
 * @param {Function} props.onLoadSessionDetails - Handler to load session details
 * @param {Function} props.onCreateSession - Handler to create new session
 * @param {Function} props.onUpdateSession - Handler to update session
 * @param {Function} props.onDeleteSession - Handler to delete session
 * @param {Function} props.onCloseSession - Handler to close active session
 * @param {Function} props.onUploadMedia - Handler to upload media file
 * @param {Function} props.onProcessMedia - Handler to process media
 */
export const SessionsPanel = ({
  sessions,
  sessionId,
  sessionDetails,
  isUploading,
  isProcessing,
  elapsedSeconds,
  onLoadSessions,
  onLoadSessionDetails,
  onCreateSession,
  onUpdateSession,
  onDeleteSession,
  onCloseSession,
  onUploadMedia,
  onProcessMedia,
}) => {
  // Create session form state
  const [isCreatingNew, setIsCreatingNew] = useState(false)
  const [title, setTitle] = useState('')
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [notes, setNotes] = useState('')
  const [isCreating, setIsCreating] = useState(false)

  // Edit session state
  const [isEditingSession, setIsEditingSession] = useState(false)
  const [editTitle, setEditTitle] = useState('')
  const [editYoutubeUrl, setEditYoutubeUrl] = useState('')
  const [editNotes, setEditNotes] = useState('')

  // Session notes editing state
  const [isEditingSessionNotes, setIsEditingSessionNotes] = useState(false)
  const [sessionNotesText, setSessionNotesText] = useState('')

  // Media upload state
  const [file, setFile] = useState(null)

  const handleCreateSession = async (event) => {
    event.preventDefault()
    setIsCreating(true)
    try {
      await onCreateSession({ title, youtube_url: youtubeUrl, notes })
      // Clear form and collapse
      setIsCreatingNew(false)
      setTitle('')
      setYoutubeUrl('')
      setNotes('')
    } catch (err) {
      // Error is handled by parent
    } finally {
      setIsCreating(false)
    }
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
    try {
      await onUpdateSession(sessionId, {
        title: editTitle.trim() || null,
        youtube_url: editYoutubeUrl.trim() || null,
        notes: editNotes.trim() || null,
      })
      setIsEditingSession(false)
    } catch (err) {
      // Error is handled by parent
    }
  }

  const handleEditSessionNotes = () => {
    setIsEditingSessionNotes(true)
    setSessionNotesText(sessionDetails?.notes || '')
  }

  const handleCancelSessionNotes = () => {
    setIsEditingSessionNotes(false)
    setSessionNotesText('')
  }

  const handleSaveSessionNotes = async () => {
    if (!sessionId) return
    try {
      await onUpdateSession(sessionId, {
        notes: sessionNotesText.trim() || null,
      })
      setIsEditingSessionNotes(false)
      setSessionNotesText('')
    } catch (err) {
      // Error is handled by parent
    }
  }

  const handleUpload = async (event) => {
    event.preventDefault()
    if (!file) return
    await onUploadMedia(sessionId, file)
    setFile(null)
  }

  const handleDeleteSession = async () => {
    if (!sessionId) return
    const confirmed = window.confirm(
      'Delete this session? This will remove all chunks and media for it.'
    )
    if (!confirmed) return
    await onDeleteSession(sessionId)
  }

  return (
    <div className={styles.sessionsTab}>
      <div className={styles.leftScroll}>
        <section className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Sessions</p>
              <h2>Your sessions</h2>
            </div>
            <button className="ghost" onClick={onLoadSessions}>
              Refresh
            </button>
          </div>

          <div className={styles.sessionList}>
            {/* New session button/form */}
            {!isCreatingNew ? (
              <button
                className={styles.createSessionBtn}
                onClick={() => setIsCreatingNew(true)}
              >
                <span className={styles.btnIcon}>+</span>
                <span>New Session</span>
              </button>
            ) : (
              <div className={styles.createSessionInline}>
                <form className="stack" onSubmit={handleCreateSession}>
                  <label className="field">
                    <span>Title</span>
                    <input
                      type="text"
                      placeholder="Scrim vs Team Blue"
                      value={title}
                      onChange={(e) => setTitle(e.target.value)}
                      autoFocus
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
                      {isCreating ? 'Creating...' : 'Create'}
                    </button>
                    <button
                      type="button"
                      className="ghost"
                      onClick={() => {
                        setIsCreatingNew(false)
                        setTitle('')
                        setYoutubeUrl('')
                        setNotes('')
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                </form>
              </div>
            )}

            {/* Empty state */}
            {sessions.length === 0 && !isCreatingNew && (
              <div className={styles.emptyState}>
                <p className={styles.emptyStateIcon}>📹</p>
                <p className="hint">No sessions yet. Create your first session to get started.</p>
              </div>
            )}

            {/* Session cards */}
            {sessions.map((session) => (
              <button
                key={session.id}
                className={`${styles.sessionCard} ${
                  session.id === sessionId ? styles.active : ''
                }`}
                onClick={() => onLoadSessionDetails(session.id)}
              >
                <div className={styles.sessionCardHeader}>
                  <div className={styles.sessionTitleRow}>
                    <strong className={styles.sessionName}>
                      {session.title || 'Untitled session'}
                    </strong>
                    <span className={`${styles.statusBadge} ${styles[session.status]}`}>
                      {formatters.getStatusDisplay(session.status)}
                    </span>
                  </div>
                  <div className={styles.sessionMetaRow}>
                    <span className={styles.sessionTime}>
                      {formatters.formatRelativeTime(session.created_at)}
                    </span>
                    {session.processing_duration_seconds != null && (
                      <span className={styles.durationChip}>
                        {formatters.formatDuration(session.processing_duration_seconds)}
                      </span>
                    )}
                  </div>
                </div>
                <div className={styles.sessionCardDetails}>
                  {session.youtube_url && (
                    <p className="hint" title={session.youtube_url}>
                      🎥 {formatters.shortenUrl(session.youtube_url)}
                    </p>
                  )}
                  {session.media_path && (
                    <p className="hint" title={session.media_path}>
                      📹 {formatters.basename(session.media_path)}
                    </p>
                  )}
                  {session.notes && (
                    <p className={styles.sessionNotesPreview}>
                      {session.notes.length > 60
                        ? session.notes.slice(0, 60) + '...'
                        : session.notes}
                    </p>
                  )}
                </div>
              </button>
            ))}
          </div>
        </section>
      </div>

      <div className={styles.leftFooter}>
        {sessionDetails && (
          <div className={styles.activeSessionPanel}>
            <div className={styles.activeSessionHeader}>
              <div className={styles.headerTitleSection}>
                <p className="eyebrow">Active Session</p>
                <div className={styles.titleWithStatus}>
                  <h2 className={styles.activeSessionTitle}>
                    {sessionDetails.title || 'Untitled session'}
                  </h2>
                  <span className={`${styles.statusBadge} ${styles[sessionDetails.status]}`}>
                    {formatters.getStatusDisplay(sessionDetails.status)}
                  </span>
                </div>
              </div>
              <button
                type="button"
                className={styles.closeBtn}
                onClick={onCloseSession}
                title="Close session"
              >
                ×
              </button>
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
                <div className="actions">
                  <button type="button" className="ghost" onClick={handleSaveSession}>
                    Save
                  </button>
                  <button
                    type="button"
                    className="ghost"
                    onClick={handleCancelEdit}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <div className={styles.activeSessionMetadata}>
                <div className={styles.metadataInfoRow}>
                  <span className={styles.metadataItem}>
                    {formatters.formatRelativeTime(sessionDetails.created_at)}
                  </span>
                  {sessionDetails.processing_duration_seconds != null && (
                    <>
                      <span className={styles.metadataSeparator}>•</span>
                      <span className={styles.metadataItem}>
                        {formatters.formatDuration(sessionDetails.processing_duration_seconds)}
                      </span>
                    </>
                  )}
                  {sessionDetails.youtube_url && (
                    <>
                      <span className={styles.metadataSeparator}>•</span>
                      <a
                        href={sessionDetails.youtube_url}
                        target="_blank"
                        rel="noreferrer"
                        className={styles.metadataLink}
                        title="Open YouTube video"
                      >
                        🎥 YouTube
                      </a>
                    </>
                  )}
                  {sessionDetails.media_path && (
                    <>
                      <span className={styles.metadataSeparator}>•</span>
                      <span className={styles.metadataItem} title={sessionDetails.media_path}>
                        📹 {formatters.basename(sessionDetails.media_path)}
                      </span>
                    </>
                  )}
                </div>

                {isEditingSessionNotes ? (
                  <div className={styles.sessionNotesEdit}>
                    <label className="field">
                      <span>Notes ({150 - sessionNotesText.length} chars left)</span>
                      <textarea
                        value={sessionNotesText}
                        onChange={(e) => setSessionNotesText(e.target.value)}
                        onBlur={handleSaveSessionNotes}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault()
                            handleSaveSessionNotes()
                          } else if (e.key === 'Escape') {
                            handleCancelSessionNotes()
                          }
                        }}
                        maxLength={150}
                        rows={3}
                        placeholder="Add session notes..."
                        autoFocus
                      />
                    </label>
                  </div>
                ) : sessionDetails.notes ? (
                  <div className={`${styles.metadataNotes} ${styles.editable}`} onClick={handleEditSessionNotes} title="Click to edit">
                    <span className={styles.notesText}>{sessionDetails.notes}</span>
                  </div>
                ) : (
                  <div className={`${styles.metadataNotesEmpty} ${styles.editable}`} onClick={handleEditSessionNotes} title="Click to add notes">
                    <span className={styles.notesTextPlaceholder}>Click to add notes...</span>
                  </div>
                )}

                {/* Upload controls */}
                <div className={styles.sessionUploadControls}>
                  {!sessionDetails.media_path && (
                    <div className={styles.uploadPrompt}>
                      <p className="hint">📤 Upload a video or audio file to get started</p>
                    </div>
                  )}
                  <form className={styles.uploadForm} onSubmit={handleUpload}>
                    <label className="field file">
                      <span>{sessionDetails.media_path ? 'Replace media' : 'Choose file'}</span>
                      <input
                        type="file"
                        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                      />
                    </label>
                    <div className={styles.uploadActions}>
                      <button type="submit" disabled={isUploading || !file}>
                        {isUploading ? 'Uploading...' : 'Upload'}
                      </button>
                      {sessionDetails.media_path && (
                        <button
                          type="button"
                          className="ghost"
                          onClick={() => onProcessMedia(sessionId)}
                          disabled={isProcessing}
                        >
                          {isProcessing
                            ? `Processing... ${formatters.formatTime(elapsedSeconds * 1000)}`
                            : 'Process'}
                        </button>
                      )}
                    </div>
                  </form>
                </div>

                <div className="actions">
                  <button
                    type="button"
                    className="ghost"
                    onClick={handleEditSession}
                  >
                    Edit
                  </button>
                  <button
                    type="button"
                    className="ghost"
                    onClick={handleDeleteSession}
                  >
                    Delete
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
