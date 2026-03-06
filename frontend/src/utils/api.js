// API Configuration
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

// Constants
export const PAGE_SIZE = 25

// Helper function to handle API responses
const handleResponse = async (res) => {
  if (!res.ok) {
    let errorMessage = `Request failed with status ${res.status}`
    try {
      const errorData = await res.json()
      errorMessage = errorData.detail || errorMessage
    } catch {
      // If response body is not JSON, use default error message
    }
    throw new Error(errorMessage)
  }
  return res.json()
}

// Sessions API

export const getSessions = async () => {
  const res = await fetch(`${API_BASE}/sessions`)
  return handleResponse(res)
}

export const getSessionById = async (id) => {
  const res = await fetch(`${API_BASE}/sessions/${id}`)
  return handleResponse(res)
}

export const createSession = async ({ title, youtube_url, notes }) => {
  const res = await fetch(`${API_BASE}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      title: title || null,
      youtube_url: youtube_url || null,
      notes: notes || null,
    }),
  })
  return handleResponse(res)
}

export const updateSession = async (id, { title, youtube_url, notes, speaker_names }) => {
  const body = {}
  if (title !== undefined) body.title = title || null
  if (youtube_url !== undefined) body.youtube_url = youtube_url || null
  if (notes !== undefined) body.notes = notes || null
  if (speaker_names !== undefined) body.speaker_names = speaker_names
  const res = await fetch(`${API_BASE}/sessions/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return handleResponse(res)
}

export const deleteSession = async (id) => {
  const res = await fetch(`${API_BASE}/sessions/${id}`, {
    method: 'DELETE',
  })
  return handleResponse(res)
}

export const uploadMedia = async (sessionId, file) => {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/media`, {
    method: 'POST',
    body: form,
  })
  return handleResponse(res)
}

export const processSession = async (sessionId) => {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/process`, {
    method: 'POST',
  })
  return handleResponse(res)
}

export const searchChunks = async (
  sessionId,
  { query, limit = 50, start_time_ms, end_time_ms, is_bookmarked }
) => {
  const res = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId || null,
      query: query || '',
      limit,
      start_time_ms: start_time_ms || null,
      end_time_ms: end_time_ms || null,
      is_bookmarked: is_bookmarked || null,
    }),
  })
  return handleResponse(res)
}

// Settings API

export const getSettings = async () => {
  const res = await fetch(`${API_BASE}/settings`)
  return handleResponse(res)
}

export const updateSettings = async (settings) => {
  const res = await fetch(`${API_BASE}/settings`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  })
  return handleResponse(res)
}

// Chunks API

export const updateChunk = async (chunkId, { text, notes, is_bookmarked }) => {
  const res = await fetch(`${API_BASE}/chunks/${chunkId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: text !== undefined ? text : undefined,
      notes: notes !== undefined ? notes : undefined,
      is_bookmarked: is_bookmarked !== undefined ? is_bookmarked : undefined,
    }),
  })
  return handleResponse(res)
}
