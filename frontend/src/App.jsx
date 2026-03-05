import { useState, useEffect, useRef } from 'react'
import { Sidebar } from './components/Sidebar'
import { SessionList } from './components/SessionList'
import { SessionDetail } from './components/SessionDetail'
import { TranscriptPanel } from './components/TranscriptPanel'
import { ExplorePanel } from './components/ExplorePanel'
import { YouTubePlayer } from './components/YouTubePlayer'
import { SessionTimeline } from './components/SessionTimeline'
import { SpeakerAssignmentModal } from './components/SpeakerAssignmentModal'
import { useAppState } from './hooks/useAppState'
import { useYouTubePlayer } from './hooks/useYouTubePlayer'

function App() {
  const [activeView, setActiveView] = useState('sessions')
  const [pageIndex, setPageIndex] = useState(0)
  const [showSpeakerModal, setShowSpeakerModal] = useState(false)
  const speakerPromptedRef = useRef(new Set())
  const returnToViewRef = useRef(null)
  const stashedChunksRef = useRef(null)

  const {
    sessions,
    sessionId,
    sessionDetails,
    chunks,
    allChunks,
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
    setChunks,
  } = useAppState()

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

  const handleNavigateToSession = async (targetSessionId, chunk) => {
    // Remember where we came from so the back button returns here
    returnToViewRef.current = activeView
    // Stash current search results so we can restore them on return
    stashedChunksRef.current = chunks
    await loadSessionDetails(targetSessionId)
    setActiveView('sessions')
    // loadVideo uses pendingSeekRef — the seek fires when the player is ready
    const url = chunk.youtube_url
    if (url) {
      const seconds = Math.floor(chunk.start_ms / 1000)
      loadVideo(url, seconds)
    }
  }

  const handleCloseSessionAndReturn = () => {
    const savedChunks = stashedChunksRef.current
    closeSession()
    if (returnToViewRef.current) {
      setActiveView(returnToViewRef.current)
      returnToViewRef.current = null
      // Restore search results that closeSession() cleared
      if (savedChunks) {
        setChunks(savedChunks)
        stashedChunksRef.current = null
      }
    }
  }

  // Session Mode: ready session with chunks → transcript panel + full-width video
  const isSessionMode = Boolean(
    sessionId && sessionDetails?.status === 'ready' && allChunks.length > 0
  )

  // Auto-prompt speaker assignment for newly processed sessions
  useEffect(() => {
    if (
      isSessionMode &&
      sessionDetails &&
      sessionDetails.speaker_names === null &&
      !speakerPromptedRef.current.has(sessionDetails.id) &&
      allChunks.some(c => c.speaker)
    ) {
      speakerPromptedRef.current.add(sessionDetails.id)
      setShowSpeakerModal(true)
    }
  }, [isSessionMode, sessionDetails?.id, sessionDetails?.speaker_names, allChunks])

  const handleSaveSpeakerNames = async (mapping) => {
    await updateSession(sessionId, { speaker_names: mapping })
    setShowSpeakerModal(false)
  }

  return (
    <div className="app-layout">
      {/* Navigation Mode: sidebar visible */}
      {!isSessionMode && (
        <Sidebar activeView={activeView} onNavigate={setActiveView} />
      )}

      {/* Session Mode: transcript panel + video/timeline */}
      {isSessionMode && (
        <>
          <TranscriptPanel
            chunks={allChunks}
            sessionDetails={sessionDetails}
            activeChunkId={activeChunkId}
            autoScrollEnabled={autoScrollEnabled}
            chunkListRef={chunkListRef}
            onTimestampClick={handleTimestampClick}
            onUpdateChunk={updateChunk}
            onCloseSession={handleCloseSessionAndReturn}
            onOpenSpeakerModal={() => setShowSpeakerModal(true)}
          />
          <main className="session-content">
            <YouTubePlayer
              videoUrl={currentVideoUrl || sessionDetails?.youtube_url}
              hasSession={true}
              fillHeight
            />
            <SessionTimeline
              chunks={allChunks}
              youtubePlayer={youtubePlayer}
              loadVideo={loadVideo}
              currentVideoUrl={currentVideoUrl || sessionDetails?.youtube_url}
              speakerNames={sessionDetails?.speaker_names}
            />
          </main>
        </>
      )}

      {/* Main area: always rendered, hidden in Session Mode */}
      <main className="main-area" style={{ display: isSessionMode ? 'none' : undefined }}>
        {activeView === 'sessions' && !sessionId && (
          <SessionList
            sessions={sessions}
            sessionId={sessionId}
            isUploading={isUploading}
            isProcessing={isProcessing}
            onLoadSessions={loadSessions}
            onLoadSessionDetails={loadSessionDetails}
            onCreateSession={createSession}
            onUploadMedia={uploadMedia}
          />
        )}

        {activeView === 'sessions' && sessionId && (
          <>
            <SessionDetail
              sessionDetails={sessionDetails}
              sessionId={sessionId}
              isUploading={isUploading}
              isProcessing={isProcessing}
              elapsedSeconds={elapsedSeconds}
              onUpdateSession={updateSession}
              onDeleteSession={deleteSession}
              onCloseSession={closeSession}
              onUploadMedia={uploadMedia}
              onProcessMedia={processMedia}
            />
            <YouTubePlayer
              videoUrl={currentVideoUrl || sessionDetails?.youtube_url}
              hasSession={Boolean(sessionId)}
            />
          </>
        )}

        {/* ExplorePanel: always mounted to preserve search state, hidden via CSS when not active */}
        <div style={{ display: activeView === 'explore' ? 'contents' : 'none' }}>
          <ExplorePanel
            sessions={sessions}
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
            onNavigateToSession={handleNavigateToSession}
          />
        </div>
      </main>

      {showSpeakerModal && (
        <SpeakerAssignmentModal
          chunks={allChunks}
          speakerNames={sessionDetails?.speaker_names}
          onSave={handleSaveSpeakerNames}
          onCancel={() => setShowSpeakerModal(false)}
          onPlaySample={handleTimestampClick}
        />
      )}
    </div>
  )
}

export default App
