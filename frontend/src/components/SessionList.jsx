import { useState, useEffect } from 'react'
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
  onDeleteSession,
  onProcessMedia,
  showConfirm,
}) => {
  const [isCreatingNew, setIsCreatingNew] = useState(false)
  const [title, setTitle] = useState('')
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [file, setFile] = useState(null)
  const [isCreating, setIsCreating] = useState(false)

  // Filter
  const [filterQuery, setFilterQuery] = useState('')

  // Sort
  const [sortKey, setSortKey] = useState('date')
  const [sortDir, setSortDir] = useState('desc')

  // Bulk selection
  const [selectedIds, setSelectedIds] = useState(new Set())
  const [isDeleting, setIsDeleting] = useState(false)

  // Context menu
  const [contextMenu, setContextMenu] = useState(null)

  // Clear selection when filter changes
  useEffect(() => setSelectedIds(new Set()), [filterQuery])

  // Close context menu on click outside or Escape
  useEffect(() => {
    if (!contextMenu) return
    const close = () => setContextMenu(null)
    const handleKey = (e) => e.key === 'Escape' && close()
    document.addEventListener('click', close)
    document.addEventListener('keydown', handleKey)
    return () => {
      document.removeEventListener('click', close)
      document.removeEventListener('keydown', handleKey)
    }
  }, [contextMenu])

  // --- Derived data ---

  const filteredSessions = filterQuery
    ? sessions.filter(s => (s.title || '').toLowerCase().includes(filterQuery.toLowerCase()))
    : sessions

  const parseAvgScore = (session) => {
    const card = formatters.parseScorecard(session.notes)
    return card ? formatters.averageScore(card) : null
  }

  const sortedSessions = [...filteredSessions].sort((a, b) => {
    let cmp = 0
    switch (sortKey) {
      case 'name':
        cmp = (a.title || '').localeCompare(b.title || '')
        break
      case 'date':
        cmp = (a.created_at || '').localeCompare(b.created_at || '')
        break
      case 'duration':
        cmp = (a.processing_duration_seconds ?? -1) - (b.processing_duration_seconds ?? -1)
        break
      case 'status':
        cmp = (a.status || '').localeCompare(b.status || '')
        break
      case 'score': {
        const sa = parseAvgScore(a)
        const sb = parseAvgScore(b)
        if (sa == null && sb == null) return 0
        if (sa == null) return 1
        if (sb == null) return -1
        cmp = (parseFloat(sa) || 0) - (parseFloat(sb) || 0)
        break
      }
    }
    return sortDir === 'asc' ? cmp : -cmp
  })

  // --- Handlers ---

  const handleCreate = async (e) => {
    e.preventDefault()
    setIsCreating(true)
    try {
      const session = await onCreateSession({
        title: title.trim() || null,
        youtube_url: youtubeUrl.trim() || null,
      })
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

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDir(key === 'name' ? 'asc' : 'desc')
    }
  }

  const handleContextMenu = (e, session) => {
    e.preventDefault()
    setContextMenu({ x: e.clientX, y: e.clientY, session })
  }

  const handleBulkDelete = () => {
    showConfirm({
      title: 'Delete sessions',
      message: `Delete ${selectedIds.size} session(s)? This cannot be undone.`,
      confirmLabel: 'Delete',
      confirmDanger: true,
      onConfirm: async () => {
        setIsDeleting(true)
        for (const id of selectedIds) {
          await onDeleteSession(id)
        }
        setSelectedIds(new Set())
        setIsDeleting(false)
        onLoadSessions()
      },
    })
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

  const renderSortArrow = (key) => {
    if (sortKey !== key) return null
    return <span className={styles.sortArrow}>{sortDir === 'asc' ? '▲' : '▼'}</span>
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

      {/* Filter bar */}
      <div className={styles.filterBar}>
        <input
          type="text"
          className={styles.filterInput}
          placeholder="Filter sessions..."
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
        />
        {filterQuery && (
          <button className={styles.clearBtn} onClick={() => setFilterQuery('')} title="Clear filter">
            ✕
          </button>
        )}
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

      {/* Column headers */}
      <div className={styles.listHeader}>
        <label className={styles.checkboxCell}>
          <input
            type="checkbox"
            className={`${styles.checkbox} ${selectedIds.size > 0 ? styles.checkboxVisible : ''}`}
            checked={selectedIds.size === sortedSessions.length && sortedSessions.length > 0}
            onChange={(e) => {
              if (e.target.checked) {
                setSelectedIds(new Set(sortedSessions.map(s => s.id)))
              } else {
                setSelectedIds(new Set())
              }
            }}
          />
        </label>
        <span className={styles.colHeader} onClick={() => handleSort('name')}>
          Name {renderSortArrow('name')}
          <span className={styles.listCount}> · {filteredSessions.length} sessions</span>
        </span>
        <span className={styles.colHeader} onClick={() => handleSort('date')}>
          Date {renderSortArrow('date')}
        </span>
        <span className={styles.colHeader} onClick={() => handleSort('duration')}>
          Duration {renderSortArrow('duration')}
        </span>
        <span className={styles.colHeader} onClick={() => handleSort('status')}>
          Status {renderSortArrow('status')}
        </span>
        <span className={styles.colHeader} onClick={() => handleSort('score')}>
          Score {renderSortArrow('score')}
        </span>
      </div>

      {/* Session rows */}
      <div className={`${styles.list} ${selectedIds.size > 0 ? styles.hasSelection : ''}`}>
        {sortedSessions.map((session) => (
          <button
            key={session.id}
            className={`${styles.row} ${session.id === sessionId ? styles.active : ''}`}
            onClick={() => onLoadSessionDetails(session.id)}
            onContextMenu={(e) => handleContextMenu(e, session)}
          >
            <label className={styles.checkboxCell} onClick={(e) => e.stopPropagation()}>
              <input
                type="checkbox"
                className={styles.checkbox}
                checked={selectedIds.has(session.id)}
                onChange={(e) => {
                  const next = new Set(selectedIds)
                  e.target.checked ? next.add(session.id) : next.delete(session.id)
                  setSelectedIds(next)
                }}
              />
            </label>
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

        {/* Bulk action bar */}
        {selectedIds.size > 0 && (
          <div className={styles.bulkBar}>
            <span className={styles.bulkCount}>{selectedIds.size} selected</span>
            <button
              className={styles.bulkDeleteBtn}
              disabled={isDeleting}
              onClick={handleBulkDelete}
            >
              {isDeleting ? 'Deleting...' : `Delete selected (${selectedIds.size})`}
            </button>
          </div>
        )}
      </div>

      {/* Context menu */}
      {contextMenu && (
        <div
          className={styles.contextMenu}
          style={{ top: contextMenu.y, left: contextMenu.x }}
          onClick={(e) => e.stopPropagation()}
        >
          <button className={styles.contextItem} onClick={() => { onLoadSessionDetails(contextMenu.session.id); setContextMenu(null) }}>
            Open
          </button>
          <button className={styles.contextItem} onClick={() => { onLoadSessionDetails(contextMenu.session.id); setContextMenu(null) }}>
            Edit
          </button>
          <button
            className={styles.contextItem}
            disabled={contextMenu.session.status === 'processing'}
            onClick={() => { onProcessMedia(contextMenu.session.id); setContextMenu(null) }}
          >
            Reprocess
          </button>
          <div className={styles.contextDivider} />
          <button
            className={`${styles.contextItem} ${styles.contextDanger}`}
            onClick={() => {
              showConfirm({
                title: 'Delete session',
                message: `Delete "${contextMenu.session.title || 'Untitled session'}"? This cannot be undone.`,
                confirmLabel: 'Delete',
                confirmDanger: true,
                onConfirm: () => {
                  onDeleteSession(contextMenu.session.id)
                  onLoadSessions()
                },
              })
              setContextMenu(null)
            }}
          >
            Delete
          </button>
        </div>
      )}
    </div>
  )
}
