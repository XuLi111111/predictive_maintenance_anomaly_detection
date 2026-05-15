/**
 * Probability chart for the Live + Monitor pages.
 *
 * Renders the full history of ticks so the user can scroll back through
 * previous seconds via the bottom Brush slider. The viewport defaults
 * to the most-recent `defaultWindow` ticks; drag the brush handles to
 * widen the visible range, drag the body to pan into the past.
 *
 * Hovering a point shows a custom tooltip with the anomaly probability,
 * state-machine tier, and the 8 sensor readings at that window's last
 * sample (Live ticks have sensors; CSV-replay ticks now do too).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Brush,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  TooltipProps,
  XAxis,
  YAxis,
} from "recharts";

import type { StreamStatus, TickFrame } from "../api/streamClient";

interface LiveChartProps {
  ticks: TickFrame[];
  threshold: number;
  /** How many of the most-recent ticks the Brush selects on first
   *  render. The user can widen it by dragging the handles. */
  defaultWindow?: number;
}

interface ChartRow {
  idx: number;
  prob: number;
  status: StreamStatus;
  sensors?: Record<string, number>;
  timestamp?: string;
}

export default function LiveChart({
  ticks,
  threshold,
  defaultWindow = 60,
}: LiveChartProps) {
  const data: ChartRow[] = useMemo(() => ticks.map((t) => ({
    idx: t.idx,
    prob: t.prob,
    status: t.status,
    sensors: t.sensors,
    timestamp: t.timestamp,
  })), [ticks]);

  // Brush viewport — controlled.
  // Default: the most recent `defaultWindow` rows.
  // Auto-follows the latest tick *unless* the user has dragged the
  // brush back into the past (heuristic: their end-index is > 2 rows
  // behind the newest row). This way real-time monitoring keeps
  // scrolling, but the moment you scroll back to look at history, the
  // chart stops fighting you.
  const [view, setView] = useState<{ start: number; end: number }>({
    start: 0, end: 0,
  });

  useEffect(() => {
    if (data.length === 0) return;
    const lastIdx = data.length - 1;
    setView((prev) => {
      const followingEnd = prev.end === 0 || lastIdx - prev.end <= 2;
      if (followingEnd) {
        return {
          start: Math.max(0, lastIdx - defaultWindow + 1),
          end: lastIdx,
        };
      }
      return prev;
    });
  }, [data.length, defaultWindow]);

  const handleBrush = useCallback(
    (e: { startIndex?: number; endIndex?: number } | null) => {
      if (!e) return;
      if (typeof e.startIndex === "number" && typeof e.endIndex === "number") {
        setView({ start: e.startIndex, end: e.endIndex });
      }
    },
    [],
  );

  // Color the line by current peak status.
  const latest = ticks[ticks.length - 1]?.status ?? "NORMAL";
  const lineColor =
    latest === "ALERT"   ? "var(--c-status-alert)"   :
    latest === "WARNING" ? "var(--c-status-warning)" :
    latest === "WATCH"   ? "var(--c-status-watch)"   :
    "var(--c-blue)";

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 30, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--c-border)" />
          <XAxis
            dataKey="idx"
            tick={{ fontSize: 11, fill: "var(--c-muted)" }}
            label={{
              value: "Window index (live)",
              position: "insideBottom",
              offset: -10,
              fontSize: 11,
              fill: "var(--c-muted)",
            }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 11, fill: "var(--c-muted)" }}
            label={{
              value: "Probability",
              angle: -90,
              position: "insideLeft",
              offset: 16,
              fontSize: 11,
              fill: "var(--c-muted)",
            }}
          />
          <ReferenceLine
            y={threshold}
            stroke="var(--c-amber)"
            strokeDasharray="4 4"
            label={{
              value: `threshold ${threshold.toFixed(2)}`,
              fill: "var(--c-amber)",
              fontSize: 10,
              position: "insideTopRight",
            }}
          />
          <Tooltip
            content={<ChartTooltip />}
            cursor={{ stroke: "var(--c-blue)", strokeWidth: 1, strokeDasharray: "2 3" }}
          />
          <Line
            type="monotone"
            dataKey="prob"
            stroke={lineColor}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 5, stroke: lineColor, strokeWidth: 2, fill: "var(--c-bg)" }}
            isAnimationActive={false}
          />
          {data.length > defaultWindow && (
            <Brush
              dataKey="idx"
              height={26}
              y={278}
              stroke="var(--c-blue)"
              fill="var(--c-blue-bg)"
              travellerWidth={10}
              startIndex={view.start}
              endIndex={view.end}
              onChange={handleBrush}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function ChartTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null;
  const row = payload[0]?.payload as ChartRow | undefined;
  if (!row) return null;

  const statusClass = `ct-status ct-status--${row.status.toLowerCase()}`;
  const hasSensors = !!row.sensors && Object.keys(row.sensors).length > 0;

  return (
    <div className="chart-tooltip" role="tooltip">
      <div className="ct-head">
        <span className="ct-idx">Window #{row.idx}</span>
        <span className={statusClass}>{row.status}</span>
      </div>
      <div className="ct-prob-row">
        <span className="ct-prob-label">P(anomaly in next 10 s)</span>
        <span className="ct-prob-value">{(row.prob * 100).toFixed(1)}%</span>
      </div>
      {hasSensors && (
        <>
          <div className="ct-divider" />
          <div className="ct-sensors-title">SENSORS AT THIS INSTANT</div>
          <table className="ct-sensors">
            <tbody>
              {Object.entries(row.sensors!).map(([name, value]) => (
                <tr key={name}>
                  <td className="ct-sensor-name">{name}</td>
                  <td className="ct-sensor-val">{value.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      {row.timestamp && (
        <div className="ct-ts">{row.timestamp.replace(/\.\d+$/, "")}</div>
      )}
    </div>
  );
}
