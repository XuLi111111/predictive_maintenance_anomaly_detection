/**
 * Live Pump page — always-on monitoring of the connected pump.
 *
 * Subscribes to /api/live/stream over WebSocket. Data is pushed into
 * the backend buffer by ``pump_simulator.py`` (or, in production, a
 * SCADA bridge that reads the PLC and POSTs to /api/live/ingest). The
 * page itself owns:
 *
 *   - which model the backend should be using
 *   - what alert threshold to apply to each window
 *   - rendering the rolling chart, four-tier status banner, and stats
 *
 * Model + threshold changes are pushed to the backend via
 * POST /api/live/config and confirmed back to all connected dashboards
 * via a {"type":"config", …} frame on the WS, so multiple browsers
 * stay in sync.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { ApiModel, listModels } from "../api/client";
import {
  LiveConnState,
  LiveHello,
  LiveQualityFrame,
  LiveSession,
  LiveStreamStatus,
  LiveTick,
  postLiveConfig,
} from "../api/liveClient";
import Footer from "../components/Footer";
import LiveChart from "../components/LiveChart";
import LiveControlBar from "../components/LiveControlBar";
import Nav from "../components/Nav";
import StatusBanner from "../components/StatusBanner";
import { useAlertAudio } from "../hooks/useAlertAudio";

const HISTORY_CAP = 600;
const THRESHOLD_DEBOUNCE_MS = 300;
// Client-side stale-data threshold. The server can't push a stale
// warning when nothing is ticking, so we track the latest tick on the
// client and flag if it goes silent.
const STALE_DATA_MS = 5000;

function dedupKey(code: string, channel: string | null): string {
  return channel ? `${code}:${channel}` : code;
}

function clearedCounterpart(code: string): string | null {
  if (code === "FROZEN_CLEARED") return "FROZEN_SENSOR";
  if (code === "OUT_OF_RANGE_CLEARED") return "OUT_OF_RANGE";
  if (code === "EVEN_SAMPLING_RESTORED") return "UNEVEN_SAMPLING";
  return null;
}

export default function Live() {
  const [conn, setConn] = useState<LiveConnState>("connecting");
  const [hello, setHello] = useState<LiveHello | null>(null);
  const [ticks, setTicks] = useState<LiveTick[]>([]);
  const [status, setStatus] = useState<LiveStreamStatus>("NORMAL");
  const [latestProb, setLatestProb] = useState<number | null>(null);

  const [models, setModels] = useState<ApiModel[]>([]);
  const [modelId, setModelId] = useState<string>("transformer");
  const [threshold, setThreshold] = useState<number>(0.5);
  const [savingConfig, setSavingConfig] = useState(false);
  const [configError, setConfigError] = useState<string | null>(null);
  const [paused, setPaused] = useState(false);

  // Data-quality state
  const [qualityWarnings, setQualityWarnings] = useState<
    Map<string, LiveQualityFrame>
  >(new Map());
  const [lastTickAt, setLastTickAt] = useState<number | null>(null);
  const [dataStale, setDataStale] = useState(false);

  const sessionRef = useRef<LiveSession | null>(null);
  const prevStatusRef = useRef<LiveStreamStatus>("NORMAL");
  const thresholdDebounceRef = useRef<number | null>(null);
  // Pause is consulted inside the WS onTick callback whose closure is
  // captured once at session creation. Mirror the state into a ref so
  // toggling pause takes effect immediately without rebuilding the WS.
  const pausedRef = useRef(false);
  // Audio cue plumbing (shared with the inline replay on the Upload page).
  const { unlock: unlockAudio, chime, startAlertLoop, stopAlertLoop } =
    useAlertAudio();

  const noTicksYet = ticks.length === 0;

  // ── Fetch model list + initial config ───────────────────────────────
  useEffect(() => {
    let cancelled = false;
    Promise.all([
      listModels(),
      fetch("/api/live/status").then((r) => r.json()),
    ])
      .then(([m, s]) => {
        if (cancelled) return;
        setModels(m);
        if (typeof s?.model_id === "string") setModelId(s.model_id);
        if (typeof s?.threshold === "number") setThreshold(s.threshold);
      })
      .catch(() => {
        /* Live page works without this — just the dropdown stays empty */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // ── WebSocket lifecycle ─────────────────────────────────────────────
  useEffect(() => {
    const session = new LiveSession({
      onState: setConn,
      onHello: setHello,
      onConfig: (cfg) => {
        // Backend (or another browser) changed the runtime config —
        // mirror it locally so the controls reflect reality.
        setModelId(cfg.model_id);
        setThreshold(cfg.threshold);
      },
      onQuality: (frame) => {
        // Server emits a frame only when state CHANGES. We translate
        // cleared / restored codes into removals; warnings into adds.
        setQualityWarnings((prev) => {
          const next = new Map(prev);
          const counterpart = clearedCounterpart(frame.code);
          if (counterpart) {
            next.delete(dedupKey(counterpart, frame.channel));
          } else {
            next.set(dedupKey(frame.code, frame.channel), frame);
          }
          return next;
        });
      },
      onTick: (t) => {
        setLastTickAt(Date.now());
        setDataStale(false);
        if (pausedRef.current) return;  // freeze chart / banner / stats
        setTicks((prev) => {
          const next = [...prev, t];
          return next.length > HISTORY_CAP ? next.slice(-HISTORY_CAP) : next;
        });
        setLatestProb(t.prob);
        setStatus(t.status);
      },
    });
    sessionRef.current = session;
    return () => {
      session.close();
      stopAlertLoop();
    };
  }, []);

  // ── Stale-data watchdog (client-side) ──────────────────────────────
  // The server can't push a warning when no samples are arriving, so
  // we watch the gap between ticks here. > STALE_DATA_MS without a tick
  // surfaces a "data stream silent" banner.
  useEffect(() => {
    if (lastTickAt === null) return;
    const id = window.setInterval(() => {
      setDataStale(Date.now() - lastTickAt > STALE_DATA_MS);
    }, 1000);
    return () => clearInterval(id);
  }, [lastTickAt]);

  // ── Poll buffer status while idle so the "N more samples" counts down ──
  useEffect(() => {
    if (!noTicksYet) return;
    let cancelled = false;
    const refresh = async () => {
      try {
        const res = await fetch("/api/live/status");
        if (!res.ok || cancelled) return;
        const data = (await res.json()) as {
          samples_in_buffer: number;
          window_size: number;
          current_status: LiveStreamStatus;
        };
        if (cancelled) return;
        setHello((prev) =>
          prev
            ? {
                ...prev,
                samples_in_buffer: data.samples_in_buffer,
                window_size: data.window_size,
              }
            : {
                type: "hello",
                samples_in_buffer: data.samples_in_buffer,
                window_size: data.window_size,
                current_status: data.current_status,
              },
        );
      } catch {
        /* transient — retry on next tick */
      }
    };
    refresh();
    const id = window.setInterval(refresh, 1500);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [noTicksYet]);

  // ── Audio cues on status promotions ────────────────────────────────
  useEffect(() => {
    const prev = prevStatusRef.current;
    const tier = (s: LiveStreamStatus) =>
      ({ NORMAL: 0, WATCH: 1, WARNING: 2, ALERT: 3 }[s]);
    if (tier(status) > tier(prev)) {
      if (status === "WARNING" || status === "ALERT") chime(880, 220);
    }
    if (status === "ALERT" && prev !== "ALERT") startAlertLoop();
    if (status !== "ALERT" && prev === "ALERT") stopAlertLoop();
    prevStatusRef.current = status;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  // First user gesture on the page unlocks the AudioContext (browser
  // autoplay policy).
  const handleUnlockAudio = () => {
    unlockAudio();
  };

  // ── Config callbacks ───────────────────────────────────────────────
  const handleChangeModel = useCallback(async (id: string) => {
    setModelId(id);
    setConfigError(null);
    setSavingConfig(true);
    try {
      const r = await postLiveConfig({ model_id: id });
      setModelId(r.model_id);
    } catch (e) {
      setConfigError(
        e instanceof Error ? e.message : "Could not switch model.",
      );
    } finally {
      setSavingConfig(false);
    }
  }, []);

  const handleChangeThreshold = useCallback((v: number) => {
    setThreshold(v);  // immediate UI feedback
    if (thresholdDebounceRef.current !== null) {
      clearTimeout(thresholdDebounceRef.current);
    }
    thresholdDebounceRef.current = window.setTimeout(async () => {
      setConfigError(null);
      try {
        await postLiveConfig({ threshold: v });
      } catch (e) {
        setConfigError(
          e instanceof Error ? e.message : "Could not update threshold.",
        );
      }
    }, THRESHOLD_DEBOUNCE_MS);
  }, []);

  const handleTogglePause = useCallback(() => {
    setPaused((p) => {
      const next = !p;
      pausedRef.current = next;
      // Stop the repeating alert beep while paused so it doesn't tone
      // forever on a frozen frame.
      if (next) stopAlertLoop();
      return next;
    });
  }, []);

  const handleClear = useCallback(() => {
    setTicks([]);
    setLatestProb(null);
    setStatus("NORMAL");
    prevStatusRef.current = "NORMAL";
    stopAlertLoop();
  }, []);

  // ── Stats ──────────────────────────────────────────────────────────
  const stats = useMemo(() => {
    const seen = ticks.length;
    const anomalies = ticks.filter((t) => t.in_anomaly_window).length;
    const peak = ticks.reduce((m, t) => Math.max(m, t.prob), 0);
    return { seen, anomalies, peak };
  }, [ticks]);

  const samplesNeeded = hello
    ? Math.max(hello.window_size - hello.samples_in_buffer, 0)
    : null;

  return (
    <>
      <Nav active="live" />

      <section className="monitor-section" onPointerDown={handleUnlockAudio}>
        <div className="breadcrumb">
          <Link className="breadcrumb-link" to="/">Home</Link>
          <span className="sep">/</span>
          <span className="current">Live pump</span>
        </div>

        <h2 className="upload-h">Live pump monitor</h2>
        <p className="upload-sub">
          Always-on view of the connected pump. Sensor samples are pushed
          to the backend by{" "}
          <code>app/scripts/pump_simulator.py</code> (replace with your
          SCADA bridge / PLC reader in production); each full 20-second
          window is scored with the selected model and streamed here in
          real time.
        </p>

        <LiveControlBar
          state={conn}
          models={models}
          modelId={modelId}
          threshold={threshold}
          onChangeModel={handleChangeModel}
          onChangeThreshold={handleChangeThreshold}
          saving={savingConfig}
          paused={paused}
          onTogglePause={handleTogglePause}
          onClear={handleClear}
          canPauseOrClear={!noTicksYet}
        />

        {configError && (
          <div className="upload-error visible">
            <span>{configError}</span>
          </div>
        )}

        {(dataStale || qualityWarnings.size > 0) && (
          <div className="quality-alerts">
            <div className="quality-alerts-head">
              <span className="quality-alerts-dot" />
              Data quality alerts
            </div>
            {dataStale && (
              <div className="quality-alert quality-alert--severe">
                <strong>Data stream silent.</strong> No samples received in the
                last {Math.floor((Date.now() - (lastTickAt ?? Date.now())) / 1000)}s.
                The simulator / SCADA bridge may have stopped — check the
                terminal where you ran the data source.
              </div>
            )}
            {[...qualityWarnings.values()].map((w) => (
              <div
                key={dedupKey(w.code, w.channel)}
                className="quality-alert"
              >
                <strong>
                  {w.code === "FROZEN_SENSOR" && "Sensor frozen"}
                  {w.code === "OUT_OF_RANGE" && "Out-of-range reading"}
                  {w.code === "UNEVEN_SAMPLING" && "Uneven sample cadence"}
                  {w.channel ? ` · ${w.channel}` : ""}
                  {": "}
                </strong>
                {w.message}
              </div>
            ))}
          </div>
        )}

        {noTicksYet && (
          <div className="live-empty">
            <div className="live-empty-icon">⏳</div>
            <div className="live-empty-title">
              {conn === "connecting" && "Connecting to the server…"}
              {conn === "waiting" && (
                samplesNeeded !== null && samplesNeeded > 0
                  ? `Warming up — ${samplesNeeded} more samples until first prediction`
                  : "Waiting for the first sensor sample"
              )}
              {conn === "disconnected" && "Disconnected — retrying"}
              {conn === "live" && "Receiving data — waiting for first window"}
            </div>
            <div className="live-empty-hint">
              If nothing happens, the simulator probably isn&apos;t running.
              Open a terminal on this machine and run:
            </div>
            <pre className="live-empty-cmd">
              python app/scripts/pump_simulator.py
            </pre>
          </div>
        )}

        {!noTicksYet && (
          <>
            <StatusBanner status={status} prob={latestProb} />
            <LiveChart ticks={ticks} threshold={threshold} />
            <div className="monitor-stats">
              <Stat label="Windows seen"   value={stats.seen.toLocaleString()} />
              <Stat label="Anomaly windows" value={stats.anomalies.toLocaleString()} />
              <Stat label="Peak prob"       value={`${(stats.peak * 100).toFixed(1)}%`} />
              <Stat label="Current state"   value={status} />
            </div>
          </>
        )}
      </section>

      <Footer />
    </>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="summary-stat">
      <span className="summary-label">{label}</span>
      <span className="summary-value">{value}</span>
    </div>
  );
}
