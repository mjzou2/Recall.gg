import { useState, useEffect } from 'react'
import styles from './SettingsView.module.css'
import * as api from '../utils/api'

const WHISPER_MODELS = [
  { value: 'tiny', label: 'tiny' },
  { value: 'base', label: 'base' },
  { value: 'small', label: 'small' },
  { value: 'medium', label: 'medium' },
  { value: 'large-v2', label: 'large-v2' },
  { value: 'turbo', label: 'turbo (quick, accurate)' },
  { value: 'large-v3', label: 'large-v3 (most accurate)' },
]

export const SettingsView = () => {
  const [whisperModel, setWhisperModel] = useState('')
  const [showSaved, setShowSaved] = useState(false)

  useEffect(() => {
    api.getSettings().then((data) => {
      setWhisperModel(data.whisper_model)
    }).catch(() => {})
  }, [])

  const handleModelChange = async (e) => {
    const newModel = e.target.value
    setWhisperModel(newModel)
    try {
      await api.updateSettings({ whisper_model: newModel })
      setShowSaved(true)
      setTimeout(() => setShowSaved(false), 2000)
    } catch {
      // Revert on error
      api.getSettings().then((data) => setWhisperModel(data.whisper_model)).catch(() => {})
    }
  }

  return (
    <div className={styles.settingsContainer}>
      <div className={styles.settingsContent}>
        <div className={styles.settingsHeader}>
          <h1 className={styles.settingsTitle}>Settings</h1>
        </div>

        <div className={styles.section}>
          <h2 className={styles.sectionTitle}>Whisper Model</h2>
          <p className={styles.sectionDescription}>
            Larger models are more accurate but slower to process.
          </p>
          <select
            className={styles.modelSelect}
            value={whisperModel}
            onChange={handleModelChange}
          >
            {WHISPER_MODELS.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
          <p className={`${styles.savedHint} ${showSaved ? styles.visible : ''}`}>
            Saved
          </p>
        </div>
      </div>
    </div>
  )
}
