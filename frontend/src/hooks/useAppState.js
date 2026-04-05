import { useState, useEffect, useRef, useCallback } from 'react'
import * as api from '../utils/api'

/**
 * Custom hook for managing application state
 * Centralizes all backend API calls and state management for sessions, chunks, and media
 *
 * @returns {Object} Application state and handler functions
 */
export const useAppState = () => {
  // Core state
  const [sessions, setSessions] = useState([])
  const [sessionId, setSessionId] = useState('')
  const [sessionDetails, setSessionDetails] = useState(null)
  const [chunks, setChunks] = useState([])
  const [allChunks, setAllChunks] = useState([])
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')

  // Loading states
  const [isUploading, setIsUploading] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)

  // Processing timer
  const [elapsedSeconds, setElapsedSeconds] = useState(0)

  // Processing step progress (from polling)
  const [processingStep, setProcessingStep] = useState(null)

  // Ref for polling interval ID (persists across renders)
  const pollingIntervalRef = useRef(null)

  // Load all sessions on mount
  useEffect(() => {
    loadSessions()
  }, [])

  // Timer for tracking elapsed processing time
  useEffect(() => {
    if (!isProcessing) {
      return
    }
    // Reset elapsed time when processing starts
    setElapsedSeconds(0)
    const interval = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1)
    }, 1000)
    // Cleanup: clear interval when processing ends or component unmounts
    return () => clearInterval(interval)
  }, [isProcessing])

  // Cleanup polling interval on unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
        pollingIntervalRef.current = null
      }
    }
  }, [])

  // Start polling for processing status updates
  const startPolling = useCallback((targetSessionId) => {
    // Clear any existing polling interval
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current)
      pollingIntervalRef.current = null
    }

    setIsProcessing(true)

    pollingIntervalRef.current = setInterval(async () => {
      try {
        const statusData = await api.getSessionStatus(targetSessionId)
        setProcessingStep(statusData)

        if (statusData.status === 'processing') {
          setSessions(prev => prev.map(s => s.id === targetSessionId ? { ...s, status: 'processing' } : s))
          setSessionDetails(prev => prev && prev.id === targetSessionId ? { ...prev, status: 'processing' } : prev)
        }

        if (statusData.status === 'ready') {
          clearInterval(pollingIntervalRef.current)
          pollingIntervalRef.current = null
          setIsProcessing(false)
          setProcessingStep(null)
          setStatus('Chunks ready')
          // Reload session data with chunks
          const data = await api.getSessionById(targetSessionId)
          setSessionDetails(data.session)
          setChunks(data.chunks || [])
          setAllChunks(data.chunks || [])
          // Refresh sessions list for updated status badges
          const sessionsData = await api.getSessions()
          setSessions(sessionsData)
        }

        if (statusData.status === 'failed') {
          clearInterval(pollingIntervalRef.current)
          pollingIntervalRef.current = null
          setIsProcessing(false)
          setProcessingStep(null)
          setError('Processing failed')
          setStatus('')
          // Refresh to show failed status
          const sessionsData = await api.getSessions()
          setSessions(sessionsData)
        }
      } catch (err) {
        // Poll errors are transient — don't stop polling on network hiccups
        console.warn('Status poll error:', err.message)
      }
    }, 2000)
  }, [])

  // Session handlers

  const loadSessions = async () => {
    try {
      const data = await api.getSessions()
      setSessions(data)
    } catch (err) {
      setError(err.message)
    }
  }

  const loadSessionDetails = async (id) => {
    setStatus('Loading session...')
    setError('')
    try {
      const data = await api.getSessionById(id)
      setSessionDetails(data.session)
      setChunks(data.chunks || [])
      setAllChunks(data.chunks || [])
      setSessionId(id)
      setStatus('Ready')
      // Resume polling if session is mid-processing
      if (data.session.status === 'queued' || data.session.status === 'processing') {
        startPolling(id)
      }
    } catch (err) {
      setError(err.message)
      setStatus('')
    }
  }

  const createSession = async ({ title, youtube_url, notes }) => {
    setError('')
    setStatus('Creating session...')
    try {
      const data = await api.createSession({ title, youtube_url, notes })
      setSessionId(data.id)
      setSessionDetails(data)
      setChunks([])
      setAllChunks([])
      setStatus('Session created')
      await loadSessions()
      return data
    } catch (err) {
      setError(err.message)
      setStatus('')
      throw err
    }
  }

  const updateSession = async (id, updates) => {
    setStatus('Saving session...')
    setError('')
    try {
      const data = await api.updateSession(id, updates)
      setSessionDetails(data)
      await loadSessions()
      setStatus('Session updated')
      return data
    } catch (err) {
      setError(err.message)
      setStatus('')
      throw err
    }
  }

  const deleteSession = async (id) => {
    setStatus('Deleting session...')
    setError('')
    try {
      await api.deleteSession(id)
      setSessionId('')
      setSessionDetails(null)
      setChunks([])
      setAllChunks([])
      await loadSessions()
      setStatus('Session deleted')
    } catch (err) {
      setError(err.message)
      setStatus('')
      throw err
    }
  }

  const closeSession = () => {
    // Stop polling if we're leaving a processing session
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current)
      pollingIntervalRef.current = null
    }
    setIsProcessing(false)
    setProcessingStep(null)
    setSessionId('')
    setSessionDetails(null)
    setChunks([])
    setAllChunks([])
  }

  // Media handlers

  const uploadMedia = async (sessionId, file) => {
    if (!sessionId) {
      setError('Create or select a session first.')
      return
    }
    if (!file) {
      setError('Choose a media file to upload.')
      return
    }
    setIsUploading(true)
    setError('')
    setStatus('Uploading media...')
    try {
      await api.uploadMedia(sessionId, file)
      await loadSessionDetails(sessionId)
      setStatus('Media uploaded')
    } catch (err) {
      setError(err.message)
      setStatus('')
    } finally {
      setIsUploading(false)
    }
  }

  const processMedia = async (sessionId) => {
    if (!sessionId) {
      setError('Create or select a session first.')
      return
    }
    setError('')
    setStatus('Queuing for processing...')
    // Optimistically update session status so it shows "queued" immediately
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, status: 'queued' } : s))
    setSessionDetails(prev => prev && prev.id === sessionId ? { ...prev, status: 'queued' } : prev)
    try {
      await api.processSession(sessionId)
      startPolling(sessionId)
    } catch (err) {
      setError(err.message)
      setStatus('')
    }
  }

  // Chunk handlers

  const searchChunks = async (sessionId, searchParams) => {
    setStatus('Searching...')
    setError('')
    try {
      const data = await api.searchChunks(sessionId, searchParams)
      setChunks(data.results || [])
      setStatus(`Search returned ${data.results?.length ?? 0} results`)
      return data
    } catch (err) {
      setError(err.message)
      setStatus('')
      throw err
    }
  }

  const updateChunk = async (chunkId, updates) => {
    try {
      const updatedChunk = await api.updateChunk(chunkId, updates)

      // Update the chunk in local state
      setChunks((prev) =>
        prev.map((c) => (c.id === updatedChunk.id ? updatedChunk : c))
      )
      setAllChunks((prev) =>
        prev.map((c) => (c.id === updatedChunk.id ? updatedChunk : c))
      )

      return updatedChunk
    } catch (err) {
      setError(err.message)
      throw err
    }
  }

  return {
    // State
    sessions,
    sessionId,
    sessionDetails,
    chunks,
    allChunks,
    isUploading,
    isProcessing,
    elapsedSeconds,
    processingStep,

    // Handlers
    loadSessions,
    loadSessionDetails,
    createSession,
    updateSession,
    deleteSession,
    closeSession,
    uploadMedia,
    processMedia,
    searchChunks,
    updateChunk,

    // State setters (for components that need direct access)
    setChunks,
  }
}
