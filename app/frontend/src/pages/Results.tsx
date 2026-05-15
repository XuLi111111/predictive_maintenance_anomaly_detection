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

import Footer from "../components/Footer";
import Nav from "../components/Nav";
import type { PredictResult } from "../api/client";

// Mock data — swap for real API call via client.ts once backend is ready
const MOCK_RESULT: PredictResult = {
  model_id: "xgb",
  probs: Array.from({ length: 200 }, (_, i) => {
    if (i < 80) return Math.max(0, 0.05 + Math.random() * 0.1);
    if (i < 120) return Math.min(1, 0.1 + (i - 80) * 0.02 + Math.random() * 0.05);
    if (i < 160) return Math.min(1, 0.85 + Math.random() * 0.1);
    return Math.max(0, 0.6 - (i - 160) * 0.02 + Math.random() * 0.05);
  }),
  anomaly_windows: [{ start_idx: 100, end_idx: 160, peak_prob: 0.94 }],
  peak_idx: 130,
  peak_prob: 0.94,
  total_windows: 200,
  fault_windows: 60,
  metrics: { recall: 0.8049, precision: 0.9944, f1: 0.8897 },
};

const MODEL_REGISTRY = [
  { id: "xgb",         name: "XGBoost",              family: "Ensemble",      recall: 0.8049, precision: 0.9944, f1: 0.8897 },
  { id: "rf",          name: "Random Forest",         family: "Tree",          recall: 0.8429, precision: 0.8635, f1: 0.8530 },
  { id: "et",          name: "Extra Trees",           family: "Tree",          recall: 0.7669, precision: 0.8989, f1: 0.8277 },
  { id: "gb",          name: "Gradient Boosting",     family: "Boosting",      recall: 0.8397, precision: 0.8811, f1: 0.8599 },
  { id: "lr",          name: "Logistic Regression",   family: "Linear",        recall: 0.8455, precision: 0.9755, f1: 0.9058 },
  { id: "knn",         name: "KNN",                   family: "Instance",      recall: 0.5544, precision: 0.8768, f1: 0.6793 },
  { id: "svm",         name: "SVM",                   family: "Kernel",        recall: 0.3690, precision: 0.9409, f1: 0.5301 },
  { id: "transformer", name: "TransformerFusionLite", family: "Deep Learning", recall: 0.0,    precision: 0.0,    f1: 0.0    },
];

export default function Results() {
  const result = MOCK_RESULT;
  const chartData = result.probs.map((prob, idx) => ({ idx, prob }));

  return (
    <>
      <Nav active="results" />

      <main className="results-main">

        {/* Probability chart */}
        <section className="results-section">
          <h2 className="results-section-title">Anomaly Probability</h2>
          <p className="results-section-sub">
            Shaded bands mark windows that exceed the alert threshold.
          </p>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 20, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--c-border)" />
                <XAxis
                  dataKey="idx"
                  tick={{ fontSize: 11, fill: "var(--c-muted)" }}
                  label={{ value: "Window", position: "insideBottom", offset: -10, fontSize: 11, fill: "var(--c-muted)" }}
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fontSize: 11, fill: "var(--c-muted)" }}
                  label={{ value: "Probability", angle: -90, position: "insideLeft", offset: 16, fontSize: 11, fill: "var(--c-muted)" }}
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
                {result.anomaly_windows.map((w, i) => (
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
                <Line type="monotone" dataKey="prob" stroke="var(--c-blue)" dot={false} strokeWidth={1.5} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* 4-stat summary panel */}
        <section className="results-section">
          <h2 className="results-section-title">Summary</h2>
          <div className="summary-grid">
            <div className="summary-stat">
              <span className="summary-label">Total Windows</span>
              <span className="summary-value">{result.total_windows}</span>
            </div>
            <div className="summary-stat">
              <span className="summary-label">Fault Windows</span>
              <span className="summary-value">{result.fault_windows}</span>
            </div>
            <div className="summary-stat">
              <span className="summary-label">Peak Probability</span>
              <span className="summary-value">{(result.peak_prob * 100).toFixed(1)}%</span>
            </div>
            <div className="summary-stat">
              <span className="summary-label">Peak Window</span>
              <span className="summary-value">#{result.peak_idx}</span>
            </div>
          </div>
        </section>

        {/* 8-model comparison grid */}
        <section className="results-section">
          <h2 className="results-section-title">Model Comparison</h2>
          <p className="results-section-sub">
            Metrics evaluated on the SKAB held-out test set.
          </p>
          <div className="model-compare-grid">
            {MODEL_REGISTRY.map((m) => (
              <div
                key={m.id}
                className={`model-compare-card${m.id === result.model_id ? " model-compare-card--active" : ""}`}
              >
                <div className="mcc-name">{m.name}</div>
                <div className="mcc-family">{m.family}</div>
                <div className="mcc-metrics">
                  <div className="mcc-metric">
                    <span className="mcc-metric-label">F1</span>
                    <span className="mcc-metric-value">{m.f1 > 0 ? m.f1.toFixed(3) : "—"}</span>
                  </div>
                  <div className="mcc-metric">
                    <span className="mcc-metric-label">Rec</span>
                    <span className="mcc-metric-value">{m.recall > 0 ? m.recall.toFixed(3) : "—"}</span>
                  </div>
                  <div className="mcc-metric">
                    <span className="mcc-metric-label">Pre</span>
                    <span className="mcc-metric-value">{m.precision > 0 ? m.precision.toFixed(3) : "—"}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* PDF download */}
        <section className="results-section results-actions">
          <a
            className="cta-p has-tooltip"
            data-tooltip="Download PDF report"
            href="/api/report/pdf"
            download
          >
            Download PDF report
          </a>
        </section>

      </main>

      <Footer />
    </>
  );
}