import styles from './Breadcrumbs.module.css'

export const Breadcrumbs = ({ sessionTitle, onNavigateBack }) => {
  return (
    <nav className={styles.breadcrumbs}>
      <button className={styles.segment} onClick={onNavigateBack}>
        Sessions
      </button>
      <span className={styles.separator}>/</span>
      <span className={styles.current}>
        {sessionTitle || 'Untitled session'}
      </span>
    </nav>
  )
}
