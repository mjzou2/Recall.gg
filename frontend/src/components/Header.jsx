import styles from './Header.module.css'

/**
 * App header with branding and optional status/error display
 *
 * @param {Object} props
 * @param {string} props.status - Current status message
 * @param {string} props.error - Current error message
 * @param {boolean} props.showStatusBox - Whether to show status/error badges
 */
export const Header = ({ status, error, showStatusBox }) => {
  return (
    <header className={styles.header}>
      <div>
        <h1 className={styles.h1}>RECALL.GG</h1>
        <p className={styles.eyebrow}>Scrim reviews made efficient</p>
      </div>
      {showStatusBox && (
        <div className={styles.status}>
          <span className={styles.badge}>{status || 'Idle'}</span>
          {error && <span className={`${styles.badge} ${styles.danger}`}>{error}</span>}
        </div>
      )}
    </header>
  )
}
