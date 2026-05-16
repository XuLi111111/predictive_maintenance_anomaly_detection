import type { StreamStatus } from "../api/streamClient";

const COPY: Record<StreamStatus, { title: string; sub: string }> = {
  NORMAL: {
    title: "Normal",
    sub: "Pump operating within expected ranges.",
  },
  WATCH: {
    title: "Watch",
    sub: "Probability rising — keep an eye on the trend.",
  },
  WARNING: {
    title: "Warning",
    sub: "Sustained elevated probability. An anomaly is likely soon.",
  },
  ALERT: {
    title: "ALERT",
    sub: "High-confidence anomaly predicted. Investigate the pump immediately.",
  },
};

export default function StatusBanner({
  status,
  prob,
}: {
  status: StreamStatus;
  prob: number | null;
}) {
  const copy = COPY[status];
  const cls = `status-banner status-${status.toLowerCase()}`;
  return (
    <div className={cls} role="status" aria-live="polite">
      <div className="status-dot" />
      <div className="status-text">
        <div className="status-title">{copy.title}</div>
        <div className="status-sub">{copy.sub}</div>
      </div>
      <div className="status-prob">
        {prob === null ? "—" : `${(prob * 100).toFixed(1)}%`}
      </div>
    </div>
  );
}
