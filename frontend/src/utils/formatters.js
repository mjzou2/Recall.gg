// Time Formatting

export const formatTime = (ms) => {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

export const parseTime = (timeStr) => {
  if (!timeStr || !timeStr.trim()) return null
  const parts = timeStr.trim().split(':')
  if (parts.length !== 2) return null
  const minutes = parseInt(parts[0], 10)
  const seconds = parseInt(parts[1], 10)
  if (isNaN(minutes) || isNaN(seconds)) return null
  if (seconds < 0 || seconds >= 60) return null
  if (minutes < 0) return null
  return (minutes * 60 + seconds) * 1000
}

export const formatDuration = (seconds) => {
  if (seconds == null) return null
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  if (mins === 0) return `${secs}s`
  return `${mins}m ${secs}s`
}

export const formatRelativeTime = (isoDate) => {
  const now = new Date()
  const date = new Date(isoDate)
  const diffMs = now - date
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`
  return new Date(isoDate).toLocaleDateString()
}

// URL Helpers

export const shortenUrl = (url) => {
  if (!url) return ''
  try {
    const parsed = new URL(url)
    const tail = parsed.pathname.split('/').filter(Boolean).pop()
    const fragment = parsed.search || parsed.hash || ''
    const suffix = tail ? `${tail}${fragment}` : fragment.replace(/^\?/, '')
    return `${parsed.hostname}${suffix ? `/${suffix}` : ''}`
  } catch {
    const trimmed = url.replace(/^https?:\/\//, '')
    const parts = trimmed.split('/')
    if (parts.length <= 1) return trimmed
    const last = parts[parts.length - 1] || parts[parts.length - 2]
    return `${parts[0]}/${last}`
  }
}

export const basename = (path) => {
  if (!path) return ''
  const normalized = path.split('?')[0]
  const parts = normalized.split('/')
  return parts[parts.length - 1] || normalized
}

export const buildYoutubeUrlWithTimestamp = (baseUrl, startMs) => {
  if (!baseUrl) return null
  const seconds = Math.floor(startMs / 1000)
  try {
    const url = new URL(baseUrl)
    url.searchParams.set('t', `${seconds}s`)
    return url.toString()
  } catch {
    return `${baseUrl}${baseUrl.includes('?') ? '&' : '?'}t=${seconds}s`
  }
}

export const extractYoutubeVideoId = (url) => {
  if (!url) return null
  try {
    const parsed = new URL(url)
    // Handle youtube.com/watch?v=VIDEO_ID
    if (parsed.hostname.includes('youtube.com') && parsed.searchParams.has('v')) {
      return parsed.searchParams.get('v')
    }
    // Handle youtu.be/VIDEO_ID
    if (parsed.hostname === 'youtu.be') {
      return parsed.pathname.slice(1) // Remove leading slash
    }
  } catch {
    // Invalid URL
  }
  return null
}

// Status Helpers

export const getStatusDisplay = (status) => {
  const map = {
    created: 'Created',
    uploaded: 'Uploaded',
    processing: 'Processing...',
    ready: 'Ready',
    failed: 'Failed'
  }
  return map[status] || status
}

// Text Helpers

export const getPreviewText = (text) => {
  if (!text) return ''
  const firstLine = text.split('\n')[0]
  if (firstLine.length > 140) return `${firstLine.slice(0, 140)}...`
  if (text.length > firstLine.length) return `${firstLine}...`
  return firstLine
}
