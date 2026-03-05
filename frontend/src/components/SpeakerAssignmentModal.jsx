import { useState, useMemo } from 'react'
import styles from './SpeakerAssignmentModal.module.css'
import { getSpeakerColor } from './SessionTimeline'
import { formatTime } from '../utils/formatters'

export const SpeakerAssignmentModal = ({
  chunks,
  speakerNames: initialSpeakerNames,
  onSave,
  onCancel,
  onPlaySample,
}) => {
  // Extract unique speakers in order of first appearance
  const speakers = useMemo(() => {
    const seen = new Set()
    const result = []
    for (const chunk of (chunks || [])) {
      const speaker = chunk.speaker
      if (speaker && !seen.has(speaker)) {
        seen.add(speaker)
        const match = speaker.match(/(\d+)/)
        const num = match ? match[1] : null
        const numericIndex = match ? parseInt(match[1], 10) : result.length
        result.push({
          label: speaker,
          num,
          displayIndex: numericIndex,
        })
      }
    }
    return result
  }, [chunks])

  // Initialize name inputs from existing speakerNames or empty
  const [names, setNames] = useState(() => {
    const init = {}
    for (const s of speakers) {
      if (s.num != null) {
        init[s.num] = initialSpeakerNames?.[s.num] || ''
      }
    }
    return init
  })

  const handleNameChange = (speakerNum, value) => {
    setNames(prev => ({ ...prev, [speakerNum]: value }))
  }

  const handleSave = () => {
    const mapping = {}
    for (const [key, value] of Object.entries(names)) {
      const trimmed = value.trim()
      if (trimmed) {
        mapping[key] = trimmed
      }
    }
    onSave(Object.keys(mapping).length > 0 ? mapping : {})
  }

  // Find a sample chunk for a given speaker (first chunk with >8 words)
  const findSampleChunk = (speakerLabel) => {
    return (chunks || []).find(
      c => c.speaker === speakerLabel && (c.text || '').split(/\s+/).length > 8
    ) || (chunks || []).find(c => c.speaker === speakerLabel)
  }

  if (speakers.length === 0) return null

  return (
    <div className={styles.backdrop} onClick={onCancel}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>
        <div className={styles.header}>
          <h3 className={styles.title}>Assign Speaker Names</h3>
          <button className={styles.closeBtn} onClick={onCancel}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M4 4L12 12M12 4L4 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </button>
        </div>

        <p className={styles.hint}>
          Use the play button to hear each speaker and identify them.
        </p>

        <div className={styles.body}>
          {speakers.map((speaker) => {
            const color = getSpeakerColor(speaker.displayIndex)
            const sample = findSampleChunk(speaker.label)

            return (
              <div key={speaker.label} className={styles.row}>
                <span
                  className={styles.colorDot}
                  style={{ background: color }}
                />
                <span className={styles.speakerLabel}>
                  S{speaker.displayIndex + 1}
                </span>
                <input
                  type="text"
                  className={styles.nameInput}
                  placeholder={`Speaker ${speaker.displayIndex + 1} name...`}
                  value={names[speaker.num] || ''}
                  onChange={e => handleNameChange(speaker.num, e.target.value)}
                  maxLength={30}
                />
                {sample && onPlaySample && (
                  <button
                    className={styles.sampleBtn}
                    onClick={() => onPlaySample(sample)}
                    title={`Play sample at ${formatTime(sample.start_ms)}`}
                  >
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                      <path d="M3 2.5L11 7L3 11.5V2.5Z" fill="currentColor"/>
                    </svg>
                  </button>
                )}
              </div>
            )
          })}
        </div>

        <div className={styles.footer}>
          <button className={styles.cancelBtn} onClick={onCancel}>Cancel</button>
          <button className={styles.saveBtn} onClick={handleSave}>Save</button>
        </div>
      </div>
    </div>
  )
}
