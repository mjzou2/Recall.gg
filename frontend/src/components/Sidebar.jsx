import styles from './Sidebar.module.css'

export const Sidebar = ({ activeView, onNavigate }) => {
  return (
    <nav className={styles.sidebar}>
      <div className={styles.logo}>
        <span className={styles.logoText}>RECALL<span className={styles.logoAccent}>.GG</span></span>
      </div>

      <div className={styles.nav}>
        <button
          className={`${styles.navItem} ${activeView === 'sessions' ? styles.active : ''}`}
          onClick={() => onNavigate('sessions')}
        >
          <svg className={styles.navIcon} viewBox="0 0 20 20" fill="currentColor">
            <path d="M2 4.75A.75.75 0 012.75 4h14.5a.75.75 0 010 1.5H2.75A.75.75 0 012 4.75zm0 10.5a.75.75 0 01.75-.75h14.5a.75.75 0 010 1.5H2.75a.75.75 0 01-.75-.75zM2 10a.75.75 0 01.75-.75h7.5a.75.75 0 010 1.5h-7.5A.75.75 0 012 10z" />
          </svg>
          Sessions
        </button>
        <button
          className={`${styles.navItem} ${activeView === 'explore' ? styles.active : ''}`}
          onClick={() => onNavigate('explore')}
        >
          <svg className={styles.navIcon} viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM2 9a7 7 0 1112.452 4.391l3.328 3.329a.75.75 0 11-1.06 1.06l-3.329-3.328A7 7 0 012 9z" clipRule="evenodd" />
          </svg>
          Explore
        </button>
      </div>

      <div className={styles.bottom}>
        <button className={styles.navItem} disabled>
          <svg className={styles.navIcon} viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-5.5-2.5a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0zM10 12a5.99 5.99 0 00-4.793 2.39A6.483 6.483 0 0010 16.5a6.483 6.483 0 004.793-2.11A5.99 5.99 0 0010 12z" clipRule="evenodd" />
          </svg>
          Account
        </button>
      </div>
    </nav>
  )
}
