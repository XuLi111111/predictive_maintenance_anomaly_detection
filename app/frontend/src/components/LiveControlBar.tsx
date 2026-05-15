/**
 * Top-of-page control bar for the Live Pump screen.
 *
 * Bundles the three things an operator needs to see / change at any
 * moment without leaving the page:
 *   1. Connection status (LIVE / WAITING / DISCONNECTED)
 *   2. Which model is producing the predictions (model dropdown)
 *   3. The alert threshold (slider)
 *
 * Changes here are applied at the backend — `LiveControlBar` is dumb,
 * it just renders props and emits callbacks. The parent Live page owns
 * the round-trip to POST /api/live/config.
 */
import type { ApiModel } from "../api/client";
import type { LiveConnState } from "../api/liveClient";
import LiveStatusBadge from "./LiveStatusBadge";

interface LiveControlBarProps {
  state: LiveConnState;
  models: ApiModel[];
  modelId: string;
  threshold: number;
  onChangeModel: (id: string) => void;
  onChangeThreshold: (v: number) => void;
  saving?: boolean;
  paused?: boolean;
  onTogglePause?: () => void;
  onClear?: () => void;
  /** Pause/Clear stay disabled until the chart actually has data. */
  canPauseOrClear?: boolean;
}

export default function LiveControlBar({
  state,
  models,
  modelId,
  threshold,
  onChangeModel,
  onChangeThreshold,
  saving,
  paused,
  onTogglePause,
  onClear,
  canPauseOrClear,
}: LiveControlBarProps) {
  const pct = Math.round(threshold * 100);
  return (
    <div className="live-control-bar">
      <LiveStatusBadge state={state} />

      <div className="lcb-field">
        <label className="lcb-label" htmlFor="lcb-model">MODEL</label>
        <select
          id="lcb-model"
          className="lcb-select"
          value={modelId}
          onChange={(e) => onChangeModel(e.target.value)}
          disabled={saving || models.length === 0}
        >
          {models.length === 0 && <option value="">loading…</option>}
          {models.map((m) => (
            <option key={m.id} value={m.id} disabled={m.unavailable}>
              {m.name}
              {m.unavailable
                ? " · coming soon"
                : m.f1 !== null
                ? ` · F1 ${m.f1.toFixed(2)}`
                : ""}
            </option>
          ))}
        </select>
      </div>

      <div className="lcb-field lcb-field--threshold">
        <label className="lcb-label" htmlFor="lcb-threshold">
          ALERT THRESHOLD
        </label>
        <div className="lcb-threshold-row">
          <input
            id="lcb-threshold"
            type="range"
            className="threshold-slider lcb-threshold-slider"
            min="0.01"
            max="0.99"
            step="0.01"
            value={threshold}
            onChange={(e) => onChangeThreshold(Number(e.target.value))}
            disabled={saving}
          />
          <div className="lcb-threshold-value">{pct}%</div>
        </div>
      </div>

      {onTogglePause && (
        <div className="lcb-actions">
          <button
            type="button"
            className={`lcb-action ${paused ? "lcb-action--resume" : ""}`}
            onClick={onTogglePause}
            disabled={!canPauseOrClear}
            title={paused
              ? "Resume — chart and stats start updating again"
              : "Pause — freeze the chart so you can inspect it (data keeps flowing in the background)"}
          >
            {paused ? "▶ Resume" : "⏸ Pause"}
          </button>
          {onClear && (
            <button
              type="button"
              className="lcb-action"
              onClick={onClear}
              disabled={!canPauseOrClear}
              title="Clear — wipe the chart history and start over from the next tick"
            >
              ↺ Clear
            </button>
          )}
        </div>
      )}
    </div>
  );
}
