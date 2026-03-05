import { useState } from 'react'
import styles from './SessionList.module.css'
import * as formatters from '../utils/formatters'

export const SessionList = ({
  sessions,
  sessionId,
  isUploading,
  isProcessing,
  onLoadSessions,
  onLoadSessionDetails,
  onCreateSession,
  onUploadMedia,
}) => {
  const [isCreatingNew, setIsCreatingNew] = useState(false)
  const [title, setTitle] = useState('')
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [file, setFile] = useState(null)
  const [isCreating, setIsCreating] = useState(false)

  const handleCreate = async (e) => {
    e.preventDefault()
    setIsCreating(true)
    try {
      const session = await onCreateSession({
        title: title.trim() || null,
        youtube_url: youtubeUrl.trim() || null,
      })
      // createSession already sets sessionId/sessionDetails in the hook.
      // uploadMedia also calls loadSessionDetails internally after upload.
      if (file && session?.id) {
        await onUploadMedia(session.id, file)
      }
      setIsCreatingNew(false)
      setTitle('')
      setYoutubeUrl('')
      setFile(null)
    } catch {
      // Error handled by parent
    } finally {
      setIsCreating(false)
    }
  }

  const handleCancel = () => {
    setIsCreatingNew(false)
    setTitle('')
    setYoutubeUrl('')
    setFile(null)
  }

  const renderScoreBadge = (session) => {
    const scorecard = formatters.parseScorecard(session.notes)
    const avg = formatters.averageScore(scorecard)
    if (avg == null) return <span className={styles.scoreEmpty}>--</span>

    const num = parseFloat(avg)
    let colorClass = styles.scoreGreen
    if (num < 4) colorClass = styles.scoreRed
    else if (num < 7) colorClass = styles.scoreYellow

    return (
      <span className={`${styles.scoreBadge} ${colorClass}`} title={
        scorecard ? Object.entries(scorecard).map(([k, v]) => `${k}: ${v}/10`).join('\n') : ''
      }>
        {avg}
      </span>
    )
  }

  return (
    <div className={styles.container}>
      <div className={styles.toolbar}>
        <div className={styles.toolbarLeft}>
          <h2 className={styles.title}>Sessions</h2>
        </div>
        <div className={styles.toolbarRight}>
          <button className={styles.refreshBtn} onClick={onLoadSessions} title="Refresh">
            <svg width="16" height="16" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M15.312 11.424a5.5 5.5 0 01-9.201 2.466l-.312-.311h2.433a.75.75 0 000-1.5H4.598a.75.75 0 00-.75.75v3.634a.75.75 0 001.5 0v-2.033l.312.311a7 7 0 0011.712-3.138.75.75 0 00-1.449-.39zm-11.023-3.848a.75.75 0 00.126.79A5.5 5.5 0 0113.9 6.1l.311.311h-2.432a.75.75 0 000 1.5h3.634a.75.75 0 00.75-.75V3.528a.75.75 0 00-1.5 0v2.033l-.311-.311A7 7 0 002.64 8.388a.75.75 0 001.45.388l.2-.8z" clipRule="evenodd" />
            </svg>
          </button>
          {!isCreatingNew && (
            <button className={styles.newBtn} onClick={() => setIsCreatingNew(true)}>
              + New Session
            </button>
          )}
        </div>
      </div>

      {isCreatingNew && (
        <div className={styles.createForm}>
          <form onSubmit={handleCreate}>
            <div className={styles.formGrid}>
              <label className={styles.formField}>
                <span className={styles.formLabel}>Title</span>
                <input
                  type="text"
                  placeholder="Scrim vs Team Blue"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  className={styles.formInput}
                  autoFocus
                />
              </label>
              <label className={styles.formField}>
                <span className={styles.formLabel}>YouTube URL</span>
                <input
                  type="url"
                  placeholder="https://youtube.com/watch?v=..."
                  value={youtubeUrl}
                  onChange={(e) => setYoutubeUrl(e.target.value)}
                  className={styles.formInput}
                />
              </label>
              <label className={styles.formField}>
                <span className={styles.formLabel}>Audio/Video file</span>
                <input
                  type="file"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                  className={styles.formFileInput}
                />
              </label>
            </div>
            <div className={styles.formActions}>
              <button type="submit" className={styles.createBtn} disabled={isCreating || isUploading}>
                {isCreating ? 'Creating...' : isUploading ? 'Uploading...' : 'Create'}
              </button>
              <button type="button" className={styles.cancelBtn} onClick={handleCancel}>
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      <div className={styles.listHeader}>
        <span className={styles.colName}>Name<span className={styles.listCount}> · {sessions.length} sessions</span></span>
        <span className={styles.colDate}>Date</span>
        <span className={styles.colDuration}>Duration</span>
        <span className={styles.colStatus}>Status</span>
        <span className={styles.colScore}>Score</span>
      </div>

      <div className={styles.list}>
        {sessions.map((session) => (
          <button
            key={session.id}
            className={`${styles.row} ${session.id === sessionId ? styles.active : ''}`}
            onClick={() => onLoadSessionDetails(session.id)}
          >
            <span className={styles.cellName}>
              {session.title || 'Untitled session'}
            </span>
            <span className={styles.cellDate}>
              {formatters.formatRelativeTime(session.created_at)}
            </span>
            <span className={styles.cellDuration}>
              {session.processing_duration_seconds != null
                ? formatters.formatDuration(session.processing_duration_seconds)
                : '--'}
            </span>
            <span className={`${styles.statusBadge} ${styles[session.status]}`}>
              {formatters.getStatusDisplay(session.status)}
            </span>
            <span className={styles.cellScore}>
              {renderScoreBadge(session)}
            </span>
          </button>
        ))}

        {sessions.length === 0 && !isCreatingNew && (
          <div className={styles.emptyState}>
            <svg className={styles.emptyIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z" />
            </svg>
            <p className={styles.emptyText}>No sessions yet. Create your first session to get started.</p>
          </div>
        )}
      </div>
    </div>
  )
}
