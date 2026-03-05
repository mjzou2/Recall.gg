import { useEffect } from 'react'
import styles from './ConfirmDialog.module.css'

export const ConfirmDialog = ({
  title,
  message,
  confirmLabel,
  confirmDanger,
  onConfirm,
  onCancel,
}) => {
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'Escape') onCancel()
    }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [onCancel])

  return (
    <div className={styles.backdrop} onClick={onCancel}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <h3 className={styles.title}>{title}</h3>
        <p className={styles.message}>{message}</p>
        <div className={styles.footer}>
          <button className={styles.cancelBtn} onClick={onCancel}>
            Cancel
          </button>
          <button
            className={confirmDanger ? styles.dangerBtn : styles.confirmBtn}
            onClick={onConfirm}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
