import { useState, useEffect } from 'react'
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
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')

  // Loading states
  const [isUploading, setIsUploading] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)

  // Processing timer
  const [elapsedSeconds, setElapsedSeconds] = useState(0)

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
      setSessionId(id)
      setStatus('Ready')
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
      await loadSessions()
      setStatus('Session deleted')
    } catch (err) {
      setError(err.message)
      setStatus('')
      throw err
    }
  }

  const closeSession = () => {
    setSessionId('')
    setSessionDetails(null)
    setChunks([])
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
    setIsProcessing(true)
    setError('')
    setStatus('Processing (transcribe + chunk)...')
    try {
      const data = await api.processSession(sessionId)
      setChunks(data.chunks || [])
      setStatus('Chunks ready')
      await loadSessionDetails(sessionId)
      await loadSessions()
    } catch (err) {
      setError(err.message)
      setStatus('')
    } finally {
      setIsProcessing(false)
    }
  }

  // Chunk handlers

  const searchChunks = async (sessionId, searchParams) => {
    if (!sessionId) return

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
    status,
    error,
    isUploading,
    isProcessing,
    elapsedSeconds,

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
    setStatus,
    setError,
    setChunks,
  }
}
