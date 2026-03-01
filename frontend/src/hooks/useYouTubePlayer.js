import { useEffect, useState, useRef, useCallback } from 'react'
import { extractYoutubeVideoId } from '../utils/formatters'
import { PAGE_SIZE } from '../utils/api'

/**
 * Custom hook for managing YouTube player and auto-scroll functionality
 *
 * @param {Object} sessionDetails - Session object containing youtube_url
 * @param {Array} chunks - Array of chunk objects
 * @param {number} pageIndex - Current page index for pagination
 * @param {Function} setPageIndex - Function to update page index
 * @returns {Object} Player state and refs
 */
export const useYouTubePlayer = (sessionDetails, chunks, pageIndex, setPageIndex) => {
  const [youtubePlayer, setYoutubePlayer] = useState(null)
  const [isYoutubeApiReady, setIsYoutubeApiReady] = useState(false)
  const [activeChunkId, setActiveChunkId] = useState(null)
  const [autoScrollEnabled, setAutoScrollEnabled] = useState(true)
  const [currentVideoUrl, setCurrentVideoUrl] = useState(null)

  const chunkListRef = useRef(null)
  const isAutoScrollingRef = useRef(false)
  const lastScrollTopRef = useRef(0)
  const pendingSeekRef = useRef(null)

  // Sync currentVideoUrl with session details when session changes
  useEffect(() => {
    setCurrentVideoUrl(sessionDetails?.youtube_url || null)
  }, [sessionDetails?.youtube_url])

  // Load YouTube iframe API
  useEffect(() => {
    // Check if API is already loaded
    if (window.YT && window.YT.Player) {
      setIsYoutubeApiReady(true)
      return
    }

    // Load the API script if not already present
    if (!window.YT) {
      const tag = document.createElement('script')
      tag.src = 'https://www.youtube.com/iframe_api'
      const firstScriptTag = document.getElementsByTagName('script')[0]
      firstScriptTag.parentNode.insertBefore(tag, firstScriptTag)
    }

    // Set up the callback for when API is ready
    window.onYouTubeIframeAPIReady = () => {
      setIsYoutubeApiReady(true)
    }
  }, [])

  // Initialize YouTube player when currentVideoUrl changes
  useEffect(() => {
    // Clean up existing player
    if (youtubePlayer && youtubePlayer.destroy) {
      try {
        youtubePlayer.destroy()
      } catch (err) {
        console.error('Error destroying existing player:', err)
      }
      setYoutubePlayer(null)
    }

    // Check if we have everything needed to create a player
    if (!currentVideoUrl || !isYoutubeApiReady) {
      return
    }

    const videoId = extractYoutubeVideoId(currentVideoUrl)
    if (!videoId) {
      console.warn('Could not extract video ID from:', currentVideoUrl)
      return
    }

    // Check if the player container exists
    const container = document.getElementById('youtube-player')
    if (!container) {
      console.warn('YouTube player container not found')
      return
    }

    // Create the player
    try {
      const player = new window.YT.Player('youtube-player', {
        videoId: videoId,
        width: '100%',
        height: '100%',
        playerVars: {
          autoplay: 0,
          modestbranding: 1,
        },
        events: {
          onReady: (event) => {
            setYoutubePlayer(event.target)
            // Handle pending seek from loadVideo call
            if (pendingSeekRef.current !== null) {
              event.target.seekTo(pendingSeekRef.current, true)
              pendingSeekRef.current = null
            }
          },
        },
      })

      // Cleanup on unmount or video change
      return () => {
        if (player && player.destroy) {
          try {
            player.destroy()
          } catch (err) {
            console.error('Error destroying player in cleanup:', err)
          }
        }
      }
    } catch (err) {
      console.error('Error creating YouTube player:', err)
    }
  }, [currentVideoUrl, isYoutubeApiReady])

  // Load a video and seek to a specific time
  const loadVideo = useCallback((youtubeUrl, seekSeconds) => {
    if (!youtubeUrl) return

    const newVideoId = extractYoutubeVideoId(youtubeUrl)
    if (!newVideoId) return

    const currentVideoId = extractYoutubeVideoId(currentVideoUrl)

    if (newVideoId === currentVideoId && youtubePlayer) {
      // Same video, just seek
      youtubePlayer.seekTo(seekSeconds, true)
    } else if (youtubePlayer && youtubePlayer.loadVideoById) {
      // Different video, player exists — switch video
      youtubePlayer.loadVideoById({ videoId: newVideoId, startSeconds: seekSeconds })
      setCurrentVideoUrl(youtubeUrl)
    } else {
      // No player yet — set URL to trigger player creation, store pending seek
      pendingSeekRef.current = seekSeconds
      setCurrentVideoUrl(youtubeUrl)
    }
  }, [currentVideoUrl, youtubePlayer])

  // Player position sync - poll current time and highlight active chunk
  useEffect(() => {
    if (!youtubePlayer || !youtubePlayer.getCurrentTime || chunks.length === 0) {
      return
    }

    const interval = setInterval(() => {
      try {
        // Check if player still exists and is valid
        if (!youtubePlayer || !youtubePlayer.getCurrentTime || !youtubePlayer.getPlayerState) {
          return
        }

        // Only update position when playing
        const playerState = youtubePlayer.getPlayerState()
        if (playerState !== window.YT?.PlayerState?.PLAYING) {
          return
        }

        const currentSeconds = youtubePlayer.getCurrentTime()
        const currentMs = Math.floor(currentSeconds * 1000)

        // Find the chunk that contains the current timestamp
        const activeChunk = chunks.find(
          (chunk) => chunk.start_ms <= currentMs && currentMs < chunk.end_ms
        )

        if (activeChunk) {
          const chunkKey =
            activeChunk.id ??
            `${activeChunk.start_ms}-${activeChunk.end_ms}-${activeChunk.text?.length ?? 0}`

          setActiveChunkId(chunkKey)

          // Auto-scroll to active chunk if enabled
          if (autoScrollEnabled) {
            // Find which page this chunk is on
            const chunkIndex = chunks.findIndex((chunk) => {
              const key =
                chunk.id ?? `${chunk.start_ms}-${chunk.end_ms}-${chunk.text?.length ?? 0}`
              return key === chunkKey
            })

            if (chunkIndex !== -1) {
              const targetPage = Math.floor(chunkIndex / PAGE_SIZE)

              // If chunk is on a different page, switch to that page first
              if (targetPage !== pageIndex) {
                setPageIndex(targetPage)
                // Wait for React to re-render the new page before scrolling
                setTimeout(() => {
                  isAutoScrollingRef.current = true
                  const chunkElement = document.querySelector(
                    `[data-chunk-key="${chunkKey}"]`
                  )
                  if (chunkElement) {
                    chunkElement.scrollIntoView({
                      behavior: 'smooth',
                      block: 'nearest',
                    })
                  }
                  setTimeout(() => {
                    isAutoScrollingRef.current = false
                  }, 1000)
                }, 100)
              } else {
                // Same page, scroll immediately
                isAutoScrollingRef.current = true
                const chunkElement = document.querySelector(
                  `[data-chunk-key="${chunkKey}"]`
                )
                if (chunkElement) {
                  chunkElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'nearest',
                  })
                }
                setTimeout(() => {
                  isAutoScrollingRef.current = false
                }, 1000)
              }
            }
          }
        }
      } catch (err) {
        // Silently catch errors to prevent crashes during player state changes
        console.error('Error in player position sync:', err)
      }
    }, 500)

    return () => clearInterval(interval)
  }, [youtubePlayer, chunks, autoScrollEnabled, pageIndex, setPageIndex])

  // Disable auto-scroll when user manually scrolls
  useEffect(() => {
    const container = chunkListRef.current
    if (!container) return

    // Initialize last scroll position
    lastScrollTopRef.current = container.scrollTop

    const handleScroll = () => {
      // Only disable if this is a user-initiated scroll, not a programmatic one
      if (!isAutoScrollingRef.current && autoScrollEnabled) {
        const scrollDistance = Math.abs(container.scrollTop - lastScrollTopRef.current)

        // Only disable if user scrolled more than 50 pixels (prevents accidental tiny scrolls)
        if (scrollDistance > 50) {
          setAutoScrollEnabled(false)
        }
      }

      // Update last scroll position
      lastScrollTopRef.current = container.scrollTop
    }

    container.addEventListener('scroll', handleScroll)
    return () => container.removeEventListener('scroll', handleScroll)
  }, [autoScrollEnabled])

  return {
    youtubePlayer,
    isYoutubeApiReady,
    chunkListRef,
    activeChunkId,
    autoScrollEnabled,
    setAutoScrollEnabled,
    currentVideoUrl,
    loadVideo,
  }
}
