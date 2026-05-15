/**
 * Playback control bar for the Live Monitor (T9).
 * Speed presets: 0.5× / 1× / 10× / 100×.
 */

interface SpeedControlProps {
  speed: number;
  paused: boolean;
  finished: boolean;
  onSetSpeed: (speed: number) => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
}

const PRESETS = [0.5, 1, 10, 100];

export default function SpeedControl({
  speed,
  paused,
  finished,
  onSetSpeed,
  onPause,
  onResume,
  onStop,
}: SpeedControlProps) {
  return (
    <div className="speed-control">
      <div className="speed-presets">
        <span className="speed-label">SPEED</span>
        {PRESETS.map((s) => (
          <button
            key={s}
            className={`speed-pill ${speed === s ? "active" : ""}`}
            onClick={() => onSetSpeed(s)}
            disabled={finished}
            type="button"
          >
            {s}×
          </button>
        ))}
      </div>
      <div className="speed-actions">
        {!finished && (
          paused ? (
            <button className="cta-s" onClick={onResume} type="button">▶ Resume</button>
          ) : (
            <button className="cta-s" onClick={onPause} type="button">⏸ Pause</button>
          )
        )}
        <button
          className="cta-s"
          onClick={onStop}
          disabled={finished}
          type="button"
        >
          ⏹ Stop
        </button>
      </div>
    </div>
  );
}
