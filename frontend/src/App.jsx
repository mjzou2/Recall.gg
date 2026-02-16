import { useState } from 'react'
import { Header } from './components/Header'
import { SessionsPanel } from './components/SessionsPanel'
import { ExplorePanel } from './components/ExplorePanel'
import { YouTubePlayer } from './components/YouTubePlayer'
import { useAppState } from './hooks/useAppState'
import { useYouTubePlayer } from './hooks/useYouTubePlayer'
import { FEATURE_FLAGS } from './utils/api'

function App() {
  const [sidebarTab, setSidebarTab] = useState('sessions')
  const [pageIndex, setPageIndex] = useState(0)

  // Get all application state and handlers
  const {
    sessions,
    sessionId,
    sessionDetails,
    chunks,
    status,
    error,
    isCreating,
    isUploading,
    isProcessing,
    elapsedSeconds,
    handleSessionSelect,
    handleCreateSession,
    handleUpdateSession,
    handleDeleteSession,
    handleUploadMedia,
    handleProcessMedia,
    handleSearch,
    handleUpdateChunk,
    handleCloseSession,
  } = useAppState()

  // Get YouTube player state and handlers
  const {
    youtubePlayer,
    isYoutubeApiReady,
    chunkListRef,
    activeChunkId,
    autoScrollEnabled,
    setAutoScrollEnabled,
  } = useYouTubePlayer(sessionDetails, chunks, pageIndex, setPageIndex)

  const handleTimestampClick = (chunk) => {
    if (youtubePlayer && youtubePlayer.seekTo) {
      const seconds = Math.floor(chunk.start_ms / 1000)
      youtubePlayer.seekTo(seconds, true)
    }
  }

  return (
    <div className="page">
      <Header
        status={status}
        error={error}
        showStatusBox={FEATURE_FLAGS.SHOW_STATUS_BOX}
      />

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
              <SessionsPanel
                sessions={sessions}
                sessionId={sessionId}
                sessionDetails={sessionDetails}
                isCreating={isCreating}
                isUploading={isUploading}
                isProcessing={isProcessing}
                elapsedSeconds={elapsedSeconds}
                onSessionSelect={handleSessionSelect}
                onCreateSession={handleCreateSession}
                onUpdateSession={handleUpdateSession}
                onDeleteSession={handleDeleteSession}
                onUploadMedia={handleUploadMedia}
                onProcessMedia={handleProcessMedia}
                onCloseSession={handleCloseSession}
              />
            )}

            {sidebarTab === 'explore' && (
              <ExplorePanel
                sessionId={sessionId}
                sessionDetails={sessionDetails}
                chunks={chunks}
                youtubePlayer={youtubePlayer}
                chunkListRef={chunkListRef}
                activeChunkId={activeChunkId}
                autoScrollEnabled={autoScrollEnabled}
                setAutoScrollEnabled={setAutoScrollEnabled}
                pageIndex={pageIndex}
                setPageIndex={setPageIndex}
                onSearch={handleSearch}
                onUpdateChunk={handleUpdateChunk}
                onTimestampClick={handleTimestampClick}
              />
            )}
          </div>
        </div>

        <div className="main-content">
          <YouTubePlayer
            videoUrl={sessionDetails?.youtube_url}
            onPlayerReady={() => {}}
            hasSession={Boolean(sessionId)}
          />
        </div>
      </section>
    </div>
  )
}

export default App
