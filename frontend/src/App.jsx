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
    isUploading,
    isProcessing,
    elapsedSeconds,
    loadSessions,
    loadSessionDetails,
    createSession,
    updateSession,
    deleteSession,
    uploadMedia,
    processMedia,
    searchChunks,
    updateChunk,
    closeSession,
  } = useAppState()

  // Get YouTube player state and handlers
  const {
    youtubePlayer,
    isYoutubeApiReady,
    chunkListRef,
    activeChunkId,
    autoScrollEnabled,
    setAutoScrollEnabled,
    currentVideoUrl,
    loadVideo,
  } = useYouTubePlayer(sessionDetails, chunks, pageIndex, setPageIndex)

  const handleTimestampClick = (chunk) => {
    const url = chunk.youtube_url || sessionDetails?.youtube_url
    if (!url) return
    const seconds = Math.floor(chunk.start_ms / 1000)
    loadVideo(url, seconds)
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
                isUploading={isUploading}
                isProcessing={isProcessing}
                elapsedSeconds={elapsedSeconds}
                onLoadSessions={loadSessions}
                onLoadSessionDetails={loadSessionDetails}
                onCreateSession={createSession}
                onUpdateSession={updateSession}
                onDeleteSession={deleteSession}
                onUploadMedia={uploadMedia}
                onProcessMedia={processMedia}
                onCloseSession={closeSession}
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
                onSearch={searchChunks}
                onReloadSession={loadSessionDetails}
                onUpdateChunk={updateChunk}
                onTimestampClick={handleTimestampClick}
              />
            )}
          </div>
        </div>

        <div className="main-content">
          <YouTubePlayer
            videoUrl={currentVideoUrl || sessionDetails?.youtube_url}
            hasSession={Boolean(sessionId)}
          />
        </div>
      </section>
    </div>
  )
}

export default App
