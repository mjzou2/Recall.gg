import { useState, useEffect } from 'react'
import styles from './MobileDisclaimer.module.css'

export const MobileDisclaimer = () => {
  const [isMobile, setIsMobile] = useState(() => window.innerWidth < 768)

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768)
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  if (!isMobile) return null

  return (
    <div className={styles.disclaimer}>
      <div className={styles.content}>
        <span className={styles.logo}>RECALL.GG</span>
        <p className={styles.message}>
          Recall.gg is designed for desktop. Please visit on a computer for the best experience.
        </p>
      </div>
    </div>
  )
}
