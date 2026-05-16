/**
 * Upload page (T5 + David's fixes for the bugs found in code review).
 *
 * Bug fixes vs Shouvik's branch:
 *  - CSV separator: try ';' first (SKAB), then ','. Original code only handled ','.
 *  - file_id: persist the upload's file_id and pass it on the navigate URL so
 *    Results / Monitor can actually run a prediction.
 *  - Dual CTA buttons: Live Monitor (real-time replay) + Batch Analysis.
 *  - Models loaded dynamically from /api/models so the transformer's
 *    `unavailable` flag is reflected automatically when T7 lands.
 */
import { ChangeEvent, DragEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import {
  ApiError,
  ApiModel,
  UploadResponse,
  downloadReport,
  listModels,
  sampleCsvUrl,
  uploadCsv,
} from "../api/client";
import {
  EndFrame,
  ErrorFrame,
  StreamSession,
  StreamStatus,
  TickFrame,
} from "../api/streamClient";
import Footer from "../components/Footer";
import LiveChart from "../components/LiveChart";
import Nav from "../components/Nav";
import SpeedControl from "../components/SpeedControl";
import StatusBanner from "../components/StatusBanner";
import { useAlertAudio } from "../hooks/useAlertAudio";

type ValidationState = "idle" | "valid" | "invalid";

interface FileValidation {
  state: ValidationState;
  message: string;
}

const SENSOR_COLUMNS = [
  "Accelerometer1RMS",
  "Accelerometer2RMS",
  "Current",
  "Pressure",
  "Temperature",
  "Thermocouple",
  "Voltage",
  "Volume Flow RateRMS",
];

const TIMESTAMP_COLUMN = "datetime";
const LABEL_COLUMNS = ["anomaly", "label"];

const SAMPLE_ROWS: Array<[string, string, string, string, string, string, string, string, string, string]> = [
  ["2020-01-01 00:00:00", "0.412", "0.398", "0.221", "0.541", "0.718", "0.336", "0.902", "0.104", "0"],
  ["2020-01-01 00:00:01", "0.419", "0.401", "0.225", "0.538", "0.722", "0.341", "0.899", "0.108", "0"],
];

function normalise(value: string): string {
  return value.trim().replace(/^['"]|['"]$/g, "").toLowerCase();
}

function parseHeader(text: string): string[] {
  const firstLine = text.split(/\r?\n/).find((line) => line.trim().length > 0) ?? "";
  // SKAB uses ';'. Some exports use ','. Pick whichever the header has more of.
  const sep = firstLine.split(";").length >= firstLine.split(",").length ? ";" : ",";
  return firstLine.split(sep).map((c) => c.trim().replace(/^['"]|['"]$/g, ""));
}

function validateHeader(columns: string[]): FileValidation {
  const lower = columns.map(normalise);
  const required = [TIMESTAMP_COLUMN, ...SENSOR_COLUMNS];
  const missing = required.filter((c) => !lower.includes(c.toLowerCase()));
  const hasLabel = LABEL_COLUMNS.some((c) => lower.includes(c.toLowerCase()));

  if (missing.length > 0) {
    return {
      state: "invalid",
      message: `Missing column(s): ${missing.join(", ")}. Please upload a SKAB-format CSV.`,
    };
  }
  return {
    state: "valid",
    message: hasLabel
      ? `Schema OK — 8 sensor columns + label detected.`
      : `Schema OK — 8 sensor columns. No label column (metrics will be hidden).`,
  };
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function pct(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(2)}%`;
}

function formatTimeRange(range: { start: string; end: string }): string {
  // Backend emits pandas string-cast timestamps, e.g. "2020-02-24 04:52:50.310000".
  // Trim trailing microseconds and trailing whitespace for compact display (FR-07).
  const trim = (s: string) => s.replace(/\.\d+$/, "").trim();
  return `${trim(range.start)} → ${trim(range.end)}`;
}

export default function Upload() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  // Tracks the most recently picked file so an in-flight upload from an
  // earlier pick can't overwrite the UI when its response arrives late.
  const latestFileRef = useRef<File | null>(null);

  const [models, setModels] = useState<ApiModel[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>("xgb");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [validation, setValidation] = useState<FileValidation>({
    state: "idle",
    message: "",
  });
  const [threshold, setThreshold] = useState(0.5);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadInfo, setUploadInfo] = useState<UploadResponse | null>(null);
  const [apiNotice, setApiNotice] = useState("");

  // ── Inline monitor state (replaces the old jump to /monitor) ────────
  const [streamActive, setStreamActive] = useState(false);
  const [ticks, setTicks] = useState<TickFrame[]>([]);
  const [streamStatus, setStreamStatus] = useState<StreamStatus>("NORMAL");
  const [latestProb, setLatestProb] = useState<number | null>(null);
  const [speed, setSpeed] = useState(1);
  const [paused, setPaused] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [summary, setSummary] = useState<EndFrame["summary"] | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [streamFileId, setStreamFileId] = useState<string | null>(null);
  const [streamModelId, setStreamModelId] = useState<string>("xgb");

  const sessionRef = useRef<StreamSession | null>(null);
  const prevStatusRef = useRef<StreamStatus>("NORMAL");
  const { unlock: unlockAudio, chime, startAlertLoop, stopAlertLoop } =
    useAlertAudio();

  const monitorFinished = summary !== null;

  // Pull the live model registry — keeps the UI in sync with the backend
  // when the transformer becomes available (T7).
  useEffect(() => {
    listModels()
      .then(setModels)
      .catch(() => {
        // Backend not up yet — that's fine for the first paint.
      });
  }, []);

  const selectedModel = useMemo(
    () => models.find((m) => m.id === selectedModelId) ?? null,
    [models, selectedModelId],
  );
  // The Run buttons need both a locally-valid file AND a successful server
  // upload (FR-07 — we only navigate once the backend has confirmed rows /
  // time range / schema). `uploadInfo` is the source of truth for the file_id.
  const isValid =
    validation.state === "valid" &&
    selectedFile !== null &&
    uploadInfo !== null &&
    !isUploading &&
    !isRunning;

  const resetFile = () => {
    latestFileRef.current = null;
    setSelectedFile(null);
    setValidation({ state: "idle", message: "" });
    setUploadInfo(null);
    setIsUploading(false);
    setApiNotice("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // Push the picked file to the backend immediately so the user sees the
  // server-side row count and time range (FR-07) before deciding which mode
  // to launch. Late responses for a stale file are dropped via latestFileRef.
  const startServerUpload = (file: File) => {
    latestFileRef.current = file;
    setUploadInfo(null);
    setIsUploading(true);
    setApiNotice("");
    uploadCsv(file)
      .then((info) => {
        if (latestFileRef.current !== file) return;
        setUploadInfo(info);
      })
      .catch((err) => {
        if (latestFileRef.current !== file) return;
        const e = err as ApiError;
        setApiNotice(e.message || "Backend rejected the file. Please retry.");
        setSelectedFile(null);
        setUploadInfo(null);
      })
      .finally(() => {
        if (latestFileRef.current === file) setIsUploading(false);
      });
  };

  const readAndValidate = (file: File) => {
    setApiNotice("");
    setUploadInfo(null);
    if (!file.name.toLowerCase().endsWith(".csv")) {
      latestFileRef.current = null;
      setSelectedFile(null);
      setValidation({ state: "invalid", message: "Please upload a .csv file." });
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const text = typeof reader.result === "string" ? reader.result : "";
      const columns = parseHeader(text);
      if (columns.length <= 1) {
        latestFileRef.current = null;
        setSelectedFile(null);
        setValidation({
          state: "invalid",
          message: "Could not detect a delimited header row. Use ';' or ','.",
        });
        return;
      }
      const next = validateHeader(columns);
      setValidation(next);
      if (next.state === "valid") {
        setSelectedFile(file);
        startServerUpload(file);
      } else {
        latestFileRef.current = null;
        setSelectedFile(null);
      }
    };
    reader.onerror = () => {
      latestFileRef.current = null;
      setSelectedFile(null);
      setValidation({ state: "invalid", message: "Could not read file. Try again." });
    };
    reader.readAsText(file.slice(0, 8192));
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const f = event.target.files?.[0];
    if (f) readAndValidate(f);
  };
  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };
  const handleDragLeave = () => setIsDragOver(false);
  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) readAndValidate(f);
  };

  // ─── Inline-monitor lifecycle ──────────────────────────────────────
  // Audio cues on status promotions. Same behaviour as the Live page —
  // both pages consume the shared `useAlertAudio` hook.
  useEffect(() => {
    const prev = prevStatusRef.current;
    const tier = (s: StreamStatus) =>
      ({ NORMAL: 0, WATCH: 1, WARNING: 2, ALERT: 3 }[s]);
    if (tier(streamStatus) > tier(prev)) {
      if (streamStatus === "WARNING" || streamStatus === "ALERT") {
        chime(880, 220);
      }
    }
    if (streamStatus === "ALERT" && prev !== "ALERT") startAlertLoop();
    if (streamStatus !== "ALERT" && prev === "ALERT") stopAlertLoop();
    prevStatusRef.current = streamStatus;
  }, [streamStatus, chime, startAlertLoop, stopAlertLoop]);

  // Cleanup on unmount: close session (the hook handles its own audio cleanup).
  useEffect(() => () => {
    sessionRef.current?.close();
  }, []);

  /** Start inline monitor (CSV replay) or jump to batch results. */
  const handleRun = (mode: "monitor" | "batch") => {
    if (!isValid || !uploadInfo) return;
    if (mode === "batch") {
      setIsRunning(true);
      const params = new URLSearchParams({
        file_id: uploadInfo.file_id,
        model_id: selectedModelId,
        threshold: threshold.toString(),
      });
      navigate(`/results?${params.toString()}`);
      return;
    }
    // monitor: spin up an inline WebSocket replay.
    sessionRef.current?.close();
    stopAlertLoop();
    setTicks([]);
    setSummary(null);
    setStreamStatus("NORMAL");
    prevStatusRef.current = "NORMAL";
    setLatestProb(null);
    setSpeed(1);
    setPaused(false);
    setStreamError(null);
    setStreamFileId(uploadInfo.file_id);
    setStreamModelId(selectedModelId);
    setStreamActive(true);
    unlockAudio();  // first user gesture — unlock the audio context

    const session = new StreamSession(
      {
        fileId: uploadInfo.file_id,
        modelId: selectedModelId,
        threshold,
        speed: 1,
      },
      {
        onTick: (t: TickFrame) => {
          setTicks((prev) => [...prev, t]);
          setLatestProb(t.prob);
          setStreamStatus(t.status);
        },
        onEnd: (e: EndFrame) => setSummary(e.summary),
        onError: (e: ErrorFrame) => setStreamError(`${e.code}: ${e.message}`),
        onClose: () => stopAlertLoop(),
      },
    );
    sessionRef.current = session;
  };

  // SpeedControl callbacks
  const handleSetSpeed = useCallback((s: number) => {
    setSpeed(s);
    sessionRef.current?.setSpeed(s);
    unlockAudio();
  }, [unlockAudio]);

  const handlePause = useCallback(() => {
    setPaused(true);
    sessionRef.current?.pause();
    unlockAudio();
  }, [unlockAudio]);

  const handleResume = useCallback(() => {
    setPaused(false);
    sessionRef.current?.resume();
  }, []);

  const handleStopMonitor = useCallback(() => {
    sessionRef.current?.stop();
    stopAlertLoop();
  }, [stopAlertLoop]);

  const handleReset = useCallback(() => {
    sessionRef.current?.close();
    sessionRef.current = null;
    stopAlertLoop();
    setStreamActive(false);
    setTicks([]);
    setSummary(null);
    setStreamStatus("NORMAL");
    prevStatusRef.current = "NORMAL";
    setLatestProb(null);
    setStreamError(null);
    setStreamFileId(null);
  }, [stopAlertLoop]);

  const handleDownloadReport = useCallback(async () => {
    if (!streamFileId) return;
    setDownloading(true);
    try {
      await downloadReport(streamFileId, streamModelId);
    } catch (err) {
      const msg = err && typeof err === "object" && "message" in err
        ? String((err as { message: unknown }).message)
        : "Download failed";
      setStreamError(msg);
    } finally {
      setDownloading(false);
    }
  }, [streamFileId, streamModelId]);

  const monitorStats = useMemo(() => {
    const seen = ticks.length;
    const above = ticks.filter((t) => t.prob >= threshold).length;
    const peak = ticks.reduce((m, t) => Math.max(m, t.prob), 0);
    return { seen, above, peak };
  }, [ticks, threshold]);

  return (
    <>
      <Nav active="upload" />

      {isRunning && (
        <div className="spinner-overlay active" role="status" aria-live="polite">
          <div className="spinner-card">
            <div className="spinner" />
            <div className="spinner-msg">Preparing model…</div>
          </div>
        </div>
      )}

      <section className="upload-section">
        <div className="breadcrumb">
          <Link className="breadcrumb-link" to="/">Home</Link>
          <span className="sep">/</span>
          <span className="current">New prediction</span>
        </div>

        <h2 className="upload-h">New prediction run</h2>
        <p className="upload-sub">
          Upload a SKAB-format CSV, pick a model, set the alert threshold,
          and choose how you want to view the result.
        </p>

        <div className="sec-eyebrow">— expected file format</div>
        <div className="sample-aside sample-aside--inline">
          <div className="sample-head">
            <div className="sample-head-text">
              <div className="aside-h">SKAB CSV format</div>
              <p className="aside-sub">
                Semicolon (<code>;</code>) separated · UTF-8 · one row per
                second of sensor data · at least <strong>30 rows</strong> required.
              </p>
            </div>
            <a className="dl-btn dl-btn--primary" href={sampleCsvUrl()}>
              ↓ Download sample CSV
            </a>
          </div>

          <div className="sample-table-wrap">
            <table className="sample-table">
              <thead>
                <tr>
                  <th>datetime</th>
                  {SENSOR_COLUMNS.map((c) => <th key={c}>{c}</th>)}
                  <th>
                    anomaly <span className="th-opt">optional</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {SAMPLE_ROWS.map((row, i) => (
                  <tr key={i}>
                    {row.map((cell, j) => (
                      <td key={j} className={j === 0 ? "sample-td-ts" : "sample-td-num"}>
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
                <tr className="sample-table-ellipsis">
                  <td colSpan={SENSOR_COLUMNS.length + 2}>
                    … more rows omitted ({SENSOR_COLUMNS.length + 2} columns total)
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="divider" />

        <div className="sec-eyebrow">— step 1 · upload your data</div>

        <div
          className={`drop-zone ${isDragOver ? "drag-over" : ""} ${isValid ? "file-loaded" : ""}`}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            accept=".csv,text/csv"
            aria-label="Upload SKAB CSV file"
            onChange={handleFileChange}
            type="file"
          />
          <div className="drop-icon">↑</div>
          <div className="drop-title">Drop your CSV here</div>
          <div className="drop-sub">or <span>browse files</span></div>
          <div className="drop-hint">CSV · SKAB FORMAT · 8 SENSOR CHANNELS · UTF-8</div>
        </div>

        {validation.state === "invalid" && (
          <div className="upload-error visible"><span>{validation.message}</span></div>
        )}
        {validation.state === "valid" && (
          <div className="upload-success visible">{validation.message}</div>
        )}

        {selectedFile && (
          <div className="file-preview visible">
            <div className="fp-icon">CSV</div>
            <div style={{ flex: 1 }}>
              <div className="fp-name">{selectedFile.name}</div>
              <div className="fp-size">{formatBytes(selectedFile.size)}</div>
              {isUploading && (
                <div className="fp-size">Validating with server…</div>
              )}
              {uploadInfo && (
                <div className="fp-size">
                  {uploadInfo.rows.toLocaleString()} rows ·{" "}
                  {formatTimeRange(uploadInfo.time_range)} ·{" "}
                  {uploadInfo.has_label
                    ? "anomaly label present"
                    : "no label — metrics will be hidden"}
                </div>
              )}
              {uploadInfo && uploadInfo.warnings.length > 0 && (
                <ul className="fp-warnings">
                  {uploadInfo.warnings.map((w, i) => (
                    <li key={i} className="fp-warning-item">{w}</li>
                  ))}
                </ul>
              )}
            </div>
            <button className="fp-remove" onClick={resetFile} type="button">
              Remove ✕
            </button>
          </div>
        )}

        {apiNotice && (
          <div className="upload-warning visible"><span>{apiNotice}</span></div>
        )}

        <div className="schema-row">
          <span className="schema-label">REQUIRED COLUMNS</span>
          <span className="pill req">datetime</span>
          {SENSOR_COLUMNS.map((c) => <span className="pill" key={c}>{c}</span>)}
          <span className="pill optional">anomaly optional</span>
        </div>

        <div className="divider" />

        <div className="sec-eyebrow">— step 2 · select a model</div>
        <div className="model-grid">
          {models.length === 0 ? (
            <div style={{ gridColumn: "1 / -1", color: "var(--c-muted)", fontSize: 12 }}>
              Loading models from /api/models…
            </div>
          ) : (
            models.map((m) => (
              <button
                key={m.id}
                className={`model-card ${selectedModelId === m.id ? "selected" : ""}`}
                onClick={() => !m.unavailable && setSelectedModelId(m.id)}
                disabled={m.unavailable}
                style={m.unavailable ? { opacity: 0.45, cursor: "not-allowed" } : undefined}
                type="button"
              >
                <div className="mc-tag">{m.family}</div>
                <div className="mc-name">{m.name}</div>
                {m.unavailable && (
                  <div style={{ fontSize: 9, color: "var(--c-hint)", marginTop: 4 }}>
                    Coming soon
                  </div>
                )}
              </button>
            ))
          )}
        </div>

        <div className="divider" />

        <div className="sec-eyebrow">— model performance summary</div>
        <div className="perf-panel">
          <div className="perf-panel-head">
            <div className="perf-panel-title">
              <span className="perf-panel-title-dot" />
              Test-set performance for selected model
            </div>
            <div className="perf-panel-active-model">
              {selectedModel?.name ?? selectedModelId}
            </div>
          </div>
          <div className="perf-metrics-grid">
            <div className="perf-metric">
              <div className="perf-metric-label">Recall</div>
              <div className="perf-metric-value">{pct(selectedModel?.recall ?? null)}</div>
            </div>
            <div className="perf-metric">
              <div className="perf-metric-label">Precision</div>
              <div className="perf-metric-value">{pct(selectedModel?.precision ?? null)}</div>
            </div>
            <div className="perf-metric">
              <div className="perf-metric-label">F1-Score</div>
              <div className="perf-metric-value">{pct(selectedModel?.f1 ?? null)}</div>
            </div>
          </div>
          <div className="perf-panel-foot">
            <span className="perf-panel-foot-dot" />
            {selectedModel?.unavailable
              ? "Pending artifact — see T7."
              : "Evaluated on the held-out SKAB test split (valve2)."}
          </div>
        </div>

        <div className="divider" />

        <div className="sec-eyebrow">— step 3 · configure and run</div>
        <div className="threshold-panel">
          <div className="threshold-head">
            <div className="threshold-title">How sure should the model be before raising an alert?</div>
            <div className="threshold-sub">
              For every 20-second window of sensor data, the model outputs a
              probability between 0% and 100% that an anomaly will follow in
              the next 10 seconds. Use this slider to set how high that
              probability needs to be before we flag the window as anomalous.
            </div>
          </div>

          <input
            aria-label="Alert threshold"
            className="threshold-slider"
            max="0.99"
            min="0.01"
            onChange={(e) => setThreshold(Number(e.target.value))}
            step="0.01"
            type="range"
            value={threshold}
          />

          <div className="threshold-ends">
            <span>← more alerts · catch every hint</span>
            <span>fewer alerts · only when obvious →</span>
          </div>

          <div className="threshold-readout">
            <div className="threshold-value-card">
              <div className="threshold-pct">{Math.round(threshold * 100)}%</div>
              <div className="threshold-raw">≥ {threshold.toFixed(2)}</div>
            </div>
            <div className="threshold-readout-text">
              <strong>Right now:</strong> the system will mark a window as
              anomalous whenever the model thinks there&apos;s at least a{" "}
              <strong>{Math.round(threshold * 100)}% chance</strong> of an
              anomaly in the next 10 seconds. Anything below that is treated as
              normal.
            </div>
          </div>

          <div className="threshold-presets">
            <span className="threshold-preset-label">QUICK PICKS</span>
            {[
              { label: "Sensitive", value: 0.3, hint: "catch more, accept noise" },
              { label: "Balanced",  value: 0.5, hint: "SRS default" },
              { label: "Tuned",     value: 0.59, hint: "best F1 for transformer", star: true },
              { label: "Conservative", value: 0.7, hint: "only act when sure" },
            ].map((p) => (
              <button
                key={p.value}
                type="button"
                className={`threshold-preset ${
                  Math.abs(threshold - p.value) < 0.005 ? "active" : ""
                }`}
                onClick={() => setThreshold(p.value)}
                title={p.hint}
              >
                {p.label} · {p.value.toFixed(2)}
                {p.star && <span className="threshold-preset-star"> ★</span>}
              </button>
            ))}
          </div>
        </div>

        {/* Dual CTA — Start in-page replay OR jump to batch PDF report */}
        <div className="run-row">
          <button
            className="run-btn run-btn--primary has-tooltip"
            data-tooltip={isValid ? "Stream results window-by-window below" : "Upload a valid CSV first"}
            disabled={!isValid}
            onClick={() => handleRun("monitor")}
            type="button"
          >
            ▶ Start replay
          </button>
          <button
            className="run-btn run-btn--secondary has-tooltip"
            data-tooltip={isValid ? "Skip the timeline and jump to the static results + PDF report" : "Upload a valid CSV first"}
            disabled={!isValid}
            onClick={() => handleRun("batch")}
            type="button"
          >
            ⚡ Skip to results + PDF
          </button>
        </div>

        {/* ── Step 4 (inline) · live timeline ─────────────────────────── */}
        {streamActive && (
          <>
            <div className="divider" />
            <div className="sec-eyebrow">— step 4 · live timeline</div>
            <p className="upload-sub">
              Replaying <code>{streamFileId?.slice(0, 8) || "—"}</code> through{" "}
              <strong>{streamModelId}</strong> at threshold {threshold.toFixed(2)}.
              Drag the slider below the chart to scrub history.
            </p>

            {streamError && (
              <div className="upload-error visible">
                <span>{streamError}</span>
              </div>
            )}

            <StatusBanner status={streamStatus} prob={latestProb} />
            <LiveChart ticks={ticks} threshold={threshold} />

            <div className="monitor-stats">
              <Stat label="Windows seen"    value={monitorStats.seen.toLocaleString()} />
              <Stat label="Anomaly windows" value={monitorStats.above.toLocaleString()} />
              <Stat label="Peak prob"       value={`${(monitorStats.peak * 100).toFixed(1)}%`} />
              <Stat label="State"           value={streamStatus} />
            </div>

            <SpeedControl
              speed={speed}
              paused={paused}
              finished={monitorFinished}
              onSetSpeed={handleSetSpeed}
              onPause={handlePause}
              onResume={handleResume}
              onStop={handleStopMonitor}
            />

            {summary && (
              <div className="monitor-end">
                <h3 className="results-section-title">Stream finished</h3>
                <p className="results-section-sub">
                  Total windows: {summary.total_windows.toLocaleString()} ·{" "}
                  Anomaly windows: {summary.fault_windows.toLocaleString()} ·{" "}
                  Peak {(summary.peak_prob * 100).toFixed(1)}% at #{summary.peak_idx}.
                </p>
                <div className="run-row">
                  <button
                    className="run-btn run-btn--primary"
                    onClick={handleDownloadReport}
                    disabled={downloading}
                    type="button"
                  >
                    {downloading ? "Building PDF…" : "↓ Download PDF report"}
                  </button>
                  <button
                    className="run-btn run-btn--secondary"
                    onClick={handleReset}
                    type="button"
                  >
                    ↺ Reset timeline
                  </button>
                </div>
              </div>
            )}
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
