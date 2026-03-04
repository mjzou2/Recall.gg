import styles from './YouTubePlayer.module.css'

/**
 * YouTube iframe player component with placeholder
 *
 * @param {Object} props
 * @param {string} props.videoUrl - YouTube video URL
 * @param {boolean} props.hasSession - Whether a session is currently selected
 */
export const YouTubePlayer = ({ videoUrl, hasSession, fillHeight }) => {
  // Always render #youtube-player div so the YT API can manage it.
  // Hide it when there's no video to prevent DOM removal crashes.
  return (
    <>
      <div
        className={`${styles.playerContainer} ${fillHeight ? styles.fillHeight : ''}`}
        style={{ display: videoUrl ? 'block' : 'none' }}
      >
        <div id="youtube-player"></div>
      </div>
      {!videoUrl && (
        <div className={styles.noPlayerMessage}>
          <p className="hint">
            {hasSession
              ? 'This session has no YouTube URL. Add one in the Sessions tab to see the embedded player.'
              : 'Select a session with a YouTube URL to view the player.'}
          </p>
        </div>
      )}
    </>
  )
}
