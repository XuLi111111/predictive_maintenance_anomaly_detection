/**
 * Results page (T6 + David's fixes from code review).
 *
 * Bug fixes vs Nafisa's branch:
 *  - Reads file_id / model_id / threshold from query string instead of mock.
 *  - Calls /api/predict for the chosen model + /api/predict/compare in parallel.
 *  - PDF download is POST + Blob (was a broken GET <a href>).
 *  - Handles metrics: null (FR-21).
 *  - Transformer card uses /api/models `unavailable` flag (no hardcoded zeros).
 */
import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  ApiModel,
  CompareResponse,
  PredictResult,
  downloadReport,
  listModels,
  runCompare,
  runPredict,
} from "../api/client";
import Footer from "../components/Footer";
import Nav from "../components/Nav";

export default function Results() {
  const [searchParams] = useSearchParams();
  const fileId = searchParams.get("file_id") ?? "";
  const modelId = searchParams.get("model_id") ?? "xgb";
  const threshold = Number(searchParams.get("threshold") ?? 0.5);

  const [primary, setPrimary] = useState<PredictResult | null>(null);
  const [compare, setCompare] = useState<CompareResponse | null>(null);
  const [models, setModels] = useState<ApiModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    if (!fileId) {
      setError("No file_id in URL — please start from the Upload page.");
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    Promise.all([
      runPredict(fileId, modelId, threshold),
      runCompare(fileId, threshold),
      listModels(),
    ])
      .then(([p, c, m]) => {
        setPrimary(p);
        setCompare(c);
        setModels(m);
      })
      .catch((err) => {
        setError(err?.message ?? "Failed to fetch prediction.");
      })
      .finally(() => setLoading(false));
  }, [fileId, modelId, threshold]);

  const chartData = useMemo(
    () => (primary?.probs ?? []).map((prob, idx) => ({ idx, prob })),
    [primary],
  );

  const handleDownload = async () => {
    if (!fileId) return;
    setDownloading(true);
    try {
      await downloadReport(fileId, modelId);
    } catch (err) {
      const msg = err && typeof err === "object" && "message" in err
        ? String((err as { message: unknown }).message)
        : "Download failed";
      setError(msg);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <>
      <Nav active="results" />

      <main className="results-main">
        {!fileId && (
          <section className="results-section">
            <h2 className="results-section-title">No file_id provided</h2>
            <p className="results-section-sub">
              Start from the <Link to="/upload">Upload page</Link>.
            </p>
          </section>
        )}

        {fileId && loading && (
          <section className="results-section">
            <p className="results-section-sub">Running prediction…</p>
          </section>
        )}

        {fileId && error && (
          <section className="results-section">
            <h2 className="results-section-title">Couldn't load results</h2>
            <p className="results-section-sub" style={{ color: "var(--c-danger)" }}>{error}</p>
            <Link to="/upload" className="dl-btn">Back to Upload</Link>
          </section>
        )}

        {primary && (
          <>
            {/* Probability chart */}
            <section className="results-section">
              <h2 className="results-section-title">Anomaly probability</h2>
              <p className="results-section-sub">
                Shaded bands mark windows where probability ≥ {threshold.toFixed(2)} threshold.
              </p>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 20, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--c-border)" />
                    <XAxis
                      dataKey="idx"
                      tick={{ fontSize: 11, fill: "var(--c-muted)" }}
                      label={{
                        value: "Window index",
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
                    <Tooltip
                      formatter={(v) => (v as number).toFixed(3)}
                      labelFormatter={(l) => `Window ${String(l)}`}
                      contentStyle={{
                        background: "var(--c-surface)",
                        border: "1px solid var(--c-border)",
                        borderRadius: "var(--r-md)",
                        fontSize: 12,
                      }}
                    />
                    {primary.anomaly_windows.map((w, i) => (
                      <ReferenceArea
                        key={i}
                        x1={w.start_idx}
                        x2={w.end_idx}
                        fill="var(--c-amber-bg)"
                        fillOpacity={0.7}
                        stroke="var(--c-amber)"
                        strokeWidth={1}
                      />
                    ))}
                    <Line
                      type="monotone"
                      dataKey="prob"
                      stroke="var(--c-blue)"
                      dot={false}
                      strokeWidth={1.5}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            {/* 4-stat summary */}
            <section className="results-section">
              <h2 className="results-section-title">Summary</h2>
              <div className="summary-grid">
                <Stat label="Total Windows" value={primary.total_windows.toLocaleString()} />
                <Stat label="Anomaly Windows" value={primary.fault_windows.toLocaleString()} />
                <Stat label="Peak Probability" value={`${(primary.peak_prob * 100).toFixed(1)}%`} />
                <Stat label="Peak Window" value={`#${primary.peak_idx}`} />
              </div>
            </section>

            {/* Performance — only when ground truth is in the file (FR-21) */}
            <section className="results-section">
              <h2 className="results-section-title">Performance on this file</h2>
              {primary.metrics ? (
                <div className="summary-grid">
                  <Stat label="Precision" value={primary.metrics.precision.toFixed(3)} />
                  <Stat label="Recall" value={primary.metrics.recall.toFixed(3)} />
                  <Stat label="F1" value={primary.metrics.f1.toFixed(3)} />
                </div>
              ) : (
                <p className="results-section-sub">
                  No ground-truth labels were detected in the uploaded file.
                  Performance metrics cannot be computed without labels (FR-21).
                </p>
              )}
            </section>

            {/* 8-model comparison */}
            <section className="results-section">
              <h2 className="results-section-title">Model comparison</h2>
              <p className="results-section-sub">
                Headline metrics from the SKAB held-out test set.
              </p>
              <div className="model-compare-grid">
                {models.map((m) => {
                  const live = compare?.models.find((r) => r.model_id === m.id) ?? null;
                  const liveF1 = live?.metrics?.f1 ?? null;
                  return (
                    <div
                      key={m.id}
                      className={`model-compare-card${m.id === primary.model_id ? " model-compare-card--active" : ""}`}
                      style={m.unavailable ? { opacity: 0.5 } : undefined}
                    >
                      <div className="mcc-name">{m.name}</div>
                      <div className="mcc-family">{m.family}</div>
                      <div className="mcc-metrics">
                        <CmpRow label="F1 (test)" v={m.f1} />
                        <CmpRow label="Recall" v={m.recall} />
                        <CmpRow label="Precision" v={m.precision} />
                        {liveF1 !== null && (
                          <CmpRow label="F1 (this file)" v={liveF1} />
                        )}
                      </div>
                      {m.unavailable && (
                        <div style={{ fontSize: 9, color: "var(--c-hint)", marginTop: 4 }}>
                          Coming soon
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>

            {/* PDF download */}
            <section className="results-section results-actions">
              <button
                className="cta-p has-tooltip"
                data-tooltip="Generate the PDF report for the active model"
                onClick={handleDownload}
                disabled={downloading}
                type="button"
              >
                {downloading ? "Building PDF…" : "Download PDF report"}
              </button>
            </section>
          </>
        )}
      </main>

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

function CmpRow({ label, v }: { label: string; v: number | null }) {
  return (
    <div className="mcc-metric">
      <span className="mcc-metric-label">{label}</span>
      <span className="mcc-metric-value">{v === null ? "—" : v.toFixed(3)}</span>
    </div>
  );
}
