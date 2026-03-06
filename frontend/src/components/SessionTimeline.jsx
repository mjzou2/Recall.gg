import { useState, useEffect, useMemo, useCallback } from 'react'
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip,
  ReferenceLine
} from 'recharts'
import styles from './SessionTimeline.module.css'
import { getSpeakerIndex, getSpeakerDisplayName, TAG_PATTERN } from '../utils/formatters'

// Speaker color palette - exported for reuse
export const SPEAKER_COLORS = [
  '#A78BFA', // purple
  '#34D399', // emerald
  '#F472B6', // pink
  '#60A5FA', // blue
  '#FBBF24', // amber
  '#FB923C', // orange
  '#2DD4BF', // teal
  '#E879F9', // fuchsia
]

export const TAG_COLORS = {
  objective_call: '#60A5FA',
  shotcall: '#34D399',
  disagreement: '#FBBF24',
  silence: '#71717A',
  tilt: '#EF4444',
}

export const TAG_LABELS = {
  objective_call: 'Objective',
  shotcall: 'Shotcall',
  disagreement: 'Disagreement',
  silence: 'Silence',
  tilt: 'Tilt',
}

export const getSpeakerColor = (index) =>
  SPEAKER_COLORS[index % SPEAKER_COLORS.length]

const formatTimeAxis = (seconds) => {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

const formatSpeakerLabel = (key, speakerNames) => {
  if (key === 'unknown') return 'Unknown'
  return getSpeakerDisplayName(key, speakerNames) || key
}

export const SessionTimeline = ({ chunks, youtubePlayer, loadVideo, currentVideoUrl, speakerNames }) => {
  const [currentTime, setCurrentTime] = useState(0)
  const [hoveredTag, setHoveredTag] = useState(null)

  const renderTooltip = useCallback(({ active, payload, label }) => {
    if (!active || !payload || payload.length === 0) return null
    return (
      <div className={styles.tooltip}>
        <div className={styles.tooltipTime}>{formatTimeAxis(label)}</div>
        {payload.map((entry, i) => (
          <div key={i} className={styles.tooltipRow}>
            <span className={styles.tooltipDot} style={{ background: entry.color }} />
            <span className={styles.tooltipLabel}>{formatSpeakerLabel(entry.dataKey, speakerNames)}</span>
            <span className={styles.tooltipValue}>{entry.value}%</span>
          </div>
        ))}
      </div>
    )
  }, [speakerNames])

  // Compute timeline data from chunks
  const { data, speakers, tagMarkers, maxPct, durationS } = useMemo(() => {
    if (!chunks || chunks.length === 0) {
      return { data: [], speakers: [], tagMarkers: [], maxPct: 100, durationS: 0 }
    }

    // Extract unique speakers in order of first appearance
    const speakerSet = new Set()
    const speakerOrder = []
    for (const chunk of chunks) {
      const speaker = chunk.speaker || 'unknown'
      if (!speakerSet.has(speaker)) {
        speakerSet.add(speaker)
        speakerOrder.push(speaker)
      }
    }

    // Session duration from max end_ms
    const durationMs = Math.max(...chunks.map(c => c.end_ms))
    const durationS = durationMs / 1000

    // Time series at 15-second intervals
    const INTERVAL = 15
    const WINDOW = 60000
    const HALF_WINDOW = WINDOW / 2
    const points = []
    let globalMaxPct = 0

    for (let t = 0; t <= durationS; t += INTERVAL) {
      const tMs = t * 1000
      const windowStart = tMs - HALF_WINDOW
      const windowEnd = tMs + HALF_WINDOW
      const point = { time_s: t }

      for (const speaker of speakerOrder) {
        let totalOverlap = 0
        for (const chunk of chunks) {
          if ((chunk.speaker || 'unknown') !== speaker) continue
          const overlapStart = Math.max(chunk.start_ms, windowStart)
          const overlapEnd = Math.min(chunk.end_ms, windowEnd)
          if (overlapEnd > overlapStart) {
            totalOverlap += (overlapEnd - overlapStart)
          }
        }
        const pct = (totalOverlap / WINDOW) * 100
        const rounded = Math.round(pct * 10) / 10
        point[speaker] = rounded
        if (rounded > globalMaxPct) globalMaxPct = rounded
      }

      points.push(point)
    }

    // Parse LLM tag markers from chunk notes
    const markers = []
    for (const chunk of chunks) {
      if (!chunk.notes) continue
      const lines = chunk.notes.split('\n')
      for (const line of lines) {
        const match = line.trim().match(TAG_PATTERN)
        if (match && TAG_COLORS[match[1]]) {
          markers.push({
            time_s: chunk.start_ms / 1000,
            tagType: match[1],
            label: match[2] || TAG_LABELS[match[1]],
          })
        }
      }
    }

    return {
      data: points,
      speakers: speakerOrder,
      tagMarkers: markers,
      maxPct: globalMaxPct,
      durationS,
    }
  }, [chunks])

  // Poll YouTube player for current time
  useEffect(() => {
    if (!youtubePlayer || !youtubePlayer.getCurrentTime) return

    const interval = setInterval(() => {
      try {
        if (!youtubePlayer.getCurrentTime || !youtubePlayer.getPlayerState) return
        const state = youtubePlayer.getPlayerState()
        if (state === window.YT?.PlayerState?.PLAYING ||
            state === window.YT?.PlayerState?.PAUSED) {
          setCurrentTime(youtubePlayer.getCurrentTime())
        }
      } catch {
        // Silently handle player errors
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [youtubePlayer])

  // Click-to-seek handler
  const handleChartClick = useCallback((event) => {
    if (!event || event.activeLabel == null) return
    const seconds = event.activeLabel
    if (currentVideoUrl) {
      loadVideo(currentVideoUrl, seconds)
    }
  }, [currentVideoUrl, loadVideo])

  if (data.length === 0) return null

  const yMax = Math.ceil((maxPct || 100) * 1.1)

  return (
    <div className={styles.container}>
      <ResponsiveContainer width="100%" height={120}>
        <LineChart data={data} onClick={handleChartClick}>
          <XAxis
            dataKey="time_s"
            tickFormatter={formatTimeAxis}
            stroke="var(--text-tertiary)"
            tick={{ fontSize: 11, fill: '#6B7280' }}
            axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
            tickLine={{ stroke: 'rgba(255,255,255,0.08)' }}
          />
          <YAxis domain={[0, yMax]} hide />
          <Tooltip content={renderTooltip} wrapperStyle={{ display: 'none' }} />

          {speakers.map((speaker) => {
            const speakerIdx = getSpeakerIndex(speaker)
            return (
            <Line
              key={speaker}
              type="monotone"
              dataKey={speaker}
              stroke={getSpeakerColor(speakerIdx)}
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 3 }}
            />
            )
          })}

          {currentTime > 0 && (
            <ReferenceLine
              x={currentTime}
              stroke="#00B8D4"
              strokeDasharray="3 3"
              strokeWidth={1.5}
            />
          )}

        </LineChart>
      </ResponsiveContainer>

      {/* Tag marker strip — HTML overlay with percentage positioning */}
      {tagMarkers.length > 0 && durationS > 0 && (
        <div
          className={styles.markerStrip}
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect()
            const pct = (e.clientX - rect.left) / rect.width
            const seconds = pct * durationS
            if (currentVideoUrl && seconds >= 0) {
              loadVideo(currentVideoUrl, Math.floor(seconds))
            }
          }}
        >
          {tagMarkers.map((marker, i) => {
            const pct = (marker.time_s / durationS) * 100
            return (
              <div
                key={`tag-${i}`}
                className={styles.markerDot}
                style={{
                  left: `${pct}%`,
                  background: TAG_COLORS[marker.tagType],
                }}
                onMouseEnter={(e) => setHoveredTag({
                  x: e.clientX,
                  y: e.clientY,
                  label: `${TAG_LABELS[marker.tagType]}: ${marker.label}`,
                  color: TAG_COLORS[marker.tagType],
                })}
                onMouseLeave={() => setHoveredTag(null)}
              />
            )
          })}
        </div>
      )}

      {/* Tag hover tooltip */}
      {hoveredTag && (
        <div
          className={styles.tagTooltip}
          style={{
            left: hoveredTag.x,
            top: hoveredTag.y - 32,
          }}
        >
          <span className={styles.tagTooltipDot} style={{ background: hoveredTag.color }} />
          {hoveredTag.label}
        </div>
      )}

      {/* Tag legend */}
      {tagMarkers.length > 0 && (
        <div className={styles.legend}>
          {Object.entries(TAG_LABELS).map(([type, label]) => {
            const count = tagMarkers.filter(m => m.tagType === type).length
            if (count === 0) return null
            return (
              <span key={type} className={styles.legendItem}>
                <span className={styles.legendDot} style={{ background: TAG_COLORS[type] }} />
                {label} ({count})
              </span>
            )
          })}
        </div>
      )}
    </div>
  )
}
