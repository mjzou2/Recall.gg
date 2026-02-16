import styles from './YouTubePlayer.module.css'

/**
 * YouTube iframe player component with placeholder
 *
 * @param {Object} props
 * @param {string} props.videoUrl - YouTube video URL
 * @param {boolean} props.hasSession - Whether a session is currently selected
 */
export const YouTubePlayer = ({ videoUrl, hasSession }) => {
  return (
    <div className={styles.mainContent}>
      {videoUrl ? (
        <div className={styles.playerContainer}>
          <div id="youtube-player"></div>
        </div>
      ) : (
        <div className={styles.noPlayerMessage}>
          <p className="hint">
            {hasSession
              ? 'This session has no YouTube URL. Add one in the Sessions tab to see the embedded player.'
              : 'Select a session with a YouTube URL to view the player.'}
          </p>
        </div>
      )}
    </div>
  )
}
