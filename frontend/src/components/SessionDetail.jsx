import { useState, useEffect, useRef } from 'react'
import styles from './SessionDetail.module.css'
import * as formatters from '../utils/formatters'

export const SessionDetail = ({
  sessionDetails,
  sessionId,
  isUploading,
  isProcessing,
  elapsedSeconds,
  onUpdateSession,
  onDeleteSession,
  onCloseSession,
  onUploadMedia,
  onProcessMedia,
  showConfirm,
  openInEditMode,
  onEditModeUsed,
}) => {
  const [isEditing, setIsEditing] = useState(false)
  const [editTitle, setEditTitle] = useState('')
  const [editYoutubeUrl, setEditYoutubeUrl] = useState('')
  const [editNotes, setEditNotes] = useState('')
  const [isEditingNotes, setIsEditingNotes] = useState(false)
  const [notesText, setNotesText] = useState('')
  const [file, setFile] = useState(null)
  const fileInputRef = useRef(null)

  useEffect(() => {
    if (openInEditMode && sessionDetails) {
      setEditTitle(sessionDetails.title || '')
      setEditYoutubeUrl(sessionDetails.youtube_url || '')
      setEditNotes((sessionDetails.notes || '').slice(0, 500))
      setIsEditing(true)
    }
  }, [openInEditMode, sessionDetails?.id])

  if (!sessionDetails) return null

  const handleEdit = () => {
    setEditTitle(sessionDetails.title || '')
    setEditYoutubeUrl(sessionDetails.youtube_url || '')
    setEditNotes((sessionDetails.notes || '').slice(0, 500))
    setIsEditing(true)
  }

  const handleCancelEdit = () => {
    setIsEditing(false)
    setEditTitle('')
    setEditYoutubeUrl('')
    setEditNotes('')
    if (openInEditMode) {
      onEditModeUsed?.()
      onCloseSession()
    }
  }

  const handleSave = async () => {
    try {
      await onUpdateSession(sessionId, {
        title: editTitle.trim() || null,
        youtube_url: editYoutubeUrl.trim() || null,
        notes: editNotes.trim() || null,
      })
      setIsEditing(false)
      if (openInEditMode) {
        onEditModeUsed?.()
        onCloseSession()
      }
    } catch {
      // Error handled by parent
    }
  }

  const handleEditNotes = () => {
    setIsEditingNotes(true)
    setNotesText(sessionDetails.notes || '')
  }

  const handleSaveNotes = async () => {
    try {
      await onUpdateSession(sessionId, {
        notes: notesText.trim() || null,
      })
      setIsEditingNotes(false)
      setNotesText('')
    } catch {
      // Error handled by parent
    }
  }

  const handleCancelNotes = () => {
    setIsEditingNotes(false)
    setNotesText('')
  }

  const handleUpload = async (e) => {
    e.preventDefault()
    if (!file) return
    await onUploadMedia(sessionId, file)
    setFile(null)
  }

  const handleDelete = () => {
    showConfirm({
      title: 'Delete session',
      message: 'Delete this session? This will remove all chunks and media.',
      confirmLabel: 'Delete',
      confirmDanger: true,
      onConfirm: () => { onEditModeUsed?.(); onDeleteSession(sessionId) },
    })
  }

  return (
    <div className={styles.container}>
      <button className={styles.backBtn} onClick={() => { onEditModeUsed?.(); onCloseSession() }}>
        <svg width="16" height="16" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M17 10a.75.75 0 01-.75.75H5.612l4.158 3.96a.75.75 0 11-1.04 1.08l-5.5-5.25a.75.75 0 010-1.08l5.5-5.25a.75.75 0 111.04 1.08L5.612 9.25H16.25A.75.75 0 0117 10z" clipRule="evenodd" />
        </svg>
        Back to Sessions
      </button>

      {isEditing ? (
        <div className={styles.editForm}>
          <label className={styles.formField}>
            <span className={styles.formLabel}>Title</span>
            <input
              type="text"
              value={editTitle}
              onChange={(e) => setEditTitle(e.target.value)}
              className={styles.formInput}
              autoFocus
            />
          </label>
          <label className={styles.formField}>
            <span className={styles.formLabel}>YouTube URL</span>
            <input
              type="url"
              value={editYoutubeUrl}
              onChange={(e) => setEditYoutubeUrl(e.target.value)}
              className={styles.formInput}
            />
          </label>
          <label className={styles.formField}>
            <span className={styles.formLabel}>
              Notes <span className={styles.charCount}>({Math.max(0, 500 - editNotes.length)} chars left)</span>
            </span>
            <textarea
              value={editNotes}
              onChange={(e) => setEditNotes(e.target.value)}
              maxLength={500}
              rows={2}
              className={styles.formTextarea}
            />
          </label>
          <div className={styles.editActions}>
            <button className={styles.saveBtn} onClick={handleSave}>Save</button>
            <button className={styles.cancelBtn} onClick={handleCancelEdit}>Cancel</button>
          </div>
        </div>
      ) : (
        <>
          <div className={styles.header}>
            <div className={styles.headerMain}>
              <h2 className={styles.sessionTitle}>
                {sessionDetails.title || 'Untitled session'}
              </h2>
              <span className={`${styles.statusBadge} ${styles[sessionDetails.status]}`}>
                {formatters.getStatusDisplay(sessionDetails.status)}
              </span>
            </div>
            <div className={styles.headerActions}>
              <button className={styles.actionBtn} onClick={handleEdit}>Edit</button>
              <button className={styles.actionBtn} onClick={handleDelete}>Delete</button>
            </div>
          </div>

          <div className={styles.metadata}>
            <span className={styles.metaItem}>
              {formatters.formatRelativeTime(sessionDetails.created_at)}
            </span>
            {sessionDetails.duration_ms != null && (
              <>
                <span className={styles.metaSep}>·</span>
                <span className={styles.metaItem}>
                  {formatters.formatTime(sessionDetails.duration_ms)}
                </span>
              </>
            )}
            {sessionDetails.processing_duration_seconds != null && (
              <>
                <span className={styles.metaSep}>·</span>
                <span className={styles.metaItem}>
                  Processed in {formatters.formatDuration(sessionDetails.processing_duration_seconds)}
                </span>
              </>
            )}
            {sessionDetails.youtube_url && (
              <>
                <span className={styles.metaSep}>·</span>
                <a
                  href={sessionDetails.youtube_url}
                  target="_blank"
                  rel="noreferrer"
                  className={styles.metaLink}
                >
                  YouTube
                </a>
              </>
            )}
            {sessionDetails.media_path && (
              <>
                <span className={styles.metaSep}>·</span>
                <span className={styles.metaItem} title={sessionDetails.media_path}>
                  {formatters.basename(sessionDetails.media_path)}
                </span>
              </>
            )}
          </div>

          {/* Notes section */}
          {isEditingNotes ? (
            <div className={styles.notesEdit}>
              <textarea
                value={notesText}
                onChange={(e) => setNotesText(e.target.value)}
                onBlur={handleSaveNotes}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handleSaveNotes()
                  } else if (e.key === 'Escape') {
                    handleCancelNotes()
                  }
                }}
                maxLength={500}
                rows={2}
                placeholder="Add session notes..."
                className={styles.notesTextarea}
                autoFocus
              />
            </div>
          ) : sessionDetails.notes ? (
            <div className={styles.notesDisplay} onClick={handleEditNotes} title="Click to edit">
              <span className={styles.notesContent}>{sessionDetails.notes}</span>
            </div>
          ) : (
            <div className={styles.notesEmpty} onClick={handleEditNotes} title="Click to add notes">
              <span className={styles.notesPlaceholder}>Click to add notes...</span>
            </div>
          )}

          {/* Upload controls */}
          <div className={styles.uploadSection}>
            {!sessionDetails.media_path && (
              <p className={styles.uploadHint}>Upload a video or audio file to get started</p>
            )}
            <form className={styles.uploadForm} onSubmit={handleUpload}>
              <div className={styles.fileField}>
                <span className={styles.formLabel}>
                  {sessionDetails.media_path ? 'Replace media' : 'Choose file'}
                </span>
                <input
                  ref={fileInputRef}
                  type="file"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                  className={styles.formFileInputHidden}
                />
                <div className={styles.fileRow}>
                  <button
                    type="button"
                    className={styles.fileChooseBtn}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Choose File
                  </button>
                  <button type="submit" className={styles.fileAccentBtn} disabled={isUploading || !file}>
                    {isUploading ? 'Uploading...' : 'Upload'}
                  </button>
                  {sessionDetails.media_path && (
                    <button
                      type="button"
                      className={styles.fileChooseBtn}
                      onClick={() => onProcessMedia(sessionId)}
                      disabled={isProcessing}
                    >
                      {isProcessing
                        ? `Processing... ${formatters.formatTime(elapsedSeconds * 1000)}`
                        : 'Process'}
                    </button>
                  )}
                  <span className={styles.fileName}>
                    {file ? file.name : !sessionDetails.media_path ? 'No file chosen' : ''}
                  </span>
                </div>
              </div>
            </form>
          </div>
        </>
      )}
    </div>
  )
}
