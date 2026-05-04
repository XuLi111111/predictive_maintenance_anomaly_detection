import { Link } from "react-router-dom";

import Footer from "../components/Footer";
import Nav from "../components/Nav";

export default function Landing() {
  return (
    <>
      <Nav active="home" />

      <section className="hero">
        <span className="eyebrow">
          <span className="eyebrow-dot" />
          Real-time anomaly detection · sliding window
        </span>
        <h1 className="headline">
          Short-term pump anomaly early warning — <em>10 seconds</em> ahead.
        </h1>
        <p className="hero-sub">
          A real-time early-warning layer for industrial centrifugal pumps. The
          system continuously analyses the past 20 seconds of sensor data to
          predict whether an anomaly will occur in the next 10 seconds — using a
          sliding window (stride 1–2 s) with trend-based warning logic.
        </p>
        <div className="cta-row" style={{ justifyContent: "center" }}>
          <Link
            className="cta-p has-tooltip"
            data-tooltip="Upload a CSV and run anomaly detection"
            to="/upload"
          >
            Try it with your data →
          </Link>
          <a className="cta-s" href="#benefits">
            See it in action
          </a>
        </div>

        <div className="mockup-wrap">
          <div className="mockup-bar">
            <div className="m-dot" />
            <div className="m-dot" />
            <div className="m-dot" />
            <div className="mockup-url">pump.detect / dashboard</div>
          </div>
          <div className="mockup-body">
            <div className="mock-head">
              <div className="mock-title">Prediction output — pump_03.csv</div>
              <div className="mock-badge">
                <span className="mock-badge-dot" />
                Anomaly detected · rising probability
              </div>
            </div>
            <svg
              className="mock-chart"
              viewBox="0 0 600 90"
              preserveAspectRatio="none"
            >
              <rect width="600" height="90" fill="#f7f7f6" />
              <rect
                x="380"
                y="0"
                width="130"
                height="90"
                fill="#fffbeb"
                opacity="0.9"
              />
              <line
                x1="380"
                y1="0"
                x2="380"
                y2="90"
                stroke="#b45309"
                strokeWidth="1"
                strokeDasharray="2,3"
              />
              <polyline
                points="0,70 30,68 60,71 90,69 120,66 150,70 180,67 210,71 240,68 270,66 300,70 330,65 360,58 380,45 400,32 420,22 440,14 460,10 480,15 500,22 520,30 540,42 560,55 580,65 600,68"
                fill="none"
                stroke="#1d4ed8"
                strokeWidth="2"
                strokeLinejoin="round"
                strokeLinecap="round"
              />
              <circle cx="460" cy="10" r="4" fill="#b45309" />
              <circle
                cx="460"
                cy="10"
                r="8"
                fill="none"
                stroke="#b45309"
                strokeWidth="1"
                opacity="0.5"
              />
            </svg>
            <div className="mock-stats">
              <div className="mock-stat">
                <div className="mock-stat-k">Confidence</div>
                <div className="mock-stat-v">94.2%</div>
              </div>
              <div className="mock-stat">
                <div className="mock-stat-k">Horizon</div>
                <div className="mock-stat-v">
                  10
                  <span style={{ fontSize: 11, fontWeight: 500, color: "#6b7280" }}>
                    {" "}
                    s
                  </span>
                </div>
              </div>
              <div className="mock-stat">
                <div className="mock-stat-k">Active model</div>
                <div className="mock-stat-v">XGBoost</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="partners">
        <span className="partners-label">In collaboration with</span>
        <div className="logo-strip">
          <div className="pm-wordmark">
            <span className="pm-dot" />
            PROGRAMMED
          </div>
          <div className="uwa-block">
            <span className="uwa-wordmark">UWA</span>
            <span className="uwa-sub">Mech. Engineering</span>
          </div>
        </div>
      </section>

      <section className="benefits" id="benefits">
        <div className="sec-eyebrow">— why it matters</div>
        <h2 className="sec-h">
          Turn sensor noise into <em>real-time early warnings</em>.
        </h2>
        <p className="sec-sub">
          Unplanned pump downtime is expensive. Using a sliding window over the
          past 20 seconds of sensor data, our system continuously predicts
          whether an anomaly will occur in the next 10 seconds — and escalates
          warnings when probability keeps rising or repeated alerts are
          detected.
        </p>
        <div className="ben-grid">
          <div className="ben">
            <div className="ben-n">01</div>
            <div className="ben-stat">
              10<span className="unit">s</span>
            </div>
            <div className="ben-label">
              Prediction horizon — each window uses 20 s of past sensor data
              with a 1–2 s stride.
            </div>
          </div>
          <div className="ben">
            <div className="ben-n">02</div>
            <div className="ben-stat">0.91</div>
            <div className="ben-label">
              Best F1 score across 8 trained models, evaluated on held-out SKAB
              data.
            </div>
          </div>
          <div className="ben">
            <div className="ben-n">03</div>
            <div className="ben-stat">
              8<span className="unit">ch</span>
            </div>
            <div className="ben-label">
              Sensor channels supported out of the box, with schema validation
              on upload.
            </div>
          </div>
        </div>
      </section>

      <section className="final">
        <h2 className="final-h">
          Ready to catch the <em>next anomaly</em> in real time?
        </h2>
        <p className="final-sub">
          Upload a CSV, pick a model, and get a sliding-window anomaly
          probability timeline in under a minute.
        </p>
        <div className="final-ctas">
          <Link className="cta-p" to="/upload">
            Start a prediction →
          </Link>
          <a className="cta-s" href="/api/sample-csv" download>
            Download sample CSV
          </a>
        </div>
      </section>

      <Footer />
    </>
  );
}
