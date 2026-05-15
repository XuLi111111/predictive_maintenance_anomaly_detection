/**
 * Connection status badge for the Live Pump page.
 *
 * Visual states match the underlying LiveConnState:
 *   - connecting   : grey, "Connecting…"
 *   - waiting      : amber, "Waiting for pump data" (socket open, no ticks)
 *   - live         : pulsing red dot, "🔴 LIVE"
 *   - disconnected : grey, "Disconnected — retrying"
 */
import type { LiveConnState } from "../api/liveClient";

const COPY: Record<LiveConnState, { label: string; hint: string }> = {
  connecting:   { label: "Connecting…",   hint: "Opening WebSocket to /api/live/stream" },
  waiting:      { label: "Waiting for data", hint: "Connected — start the simulator to begin streaming" },
  live:         { label: "LIVE",          hint: "Receiving sensor data in real time" },
  disconnected: { label: "Disconnected",  hint: "Lost connection — retrying every few seconds" },
};

export default function LiveStatusBadge({ state }: { state: LiveConnState }) {
  const copy = COPY[state];
  return (
    <div className={`live-badge live-badge--${state}`} title={copy.hint}>
      <span className="live-badge-dot" />
      <span className="live-badge-text">{copy.label}</span>
    </div>
  );
}
