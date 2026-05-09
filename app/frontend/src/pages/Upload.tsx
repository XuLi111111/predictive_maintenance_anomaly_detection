import { ChangeEvent, DragEvent, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { sampleCsvUrl, uploadCsv } from "../api/client";
import Footer from "../components/Footer";
import Nav from "../components/Nav";

type ModelId = "xgb" | "rf" | "et" | "gb" | "lr" | "knn" | "svm" | "transformer";
type ValidationState = "idle" | "valid" | "invalid";

interface ModelOption {
  id: ModelId;
  name: string;
  shortName: string;
  family: string;
  recall: number | null;
  precision: number | null;
  f1: number | null;
  note?: string;
}

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

const SAMPLE_CSV = `datetime,Accelerometer1RMS,Accelerometer2RMS,Current,Pressure,Temperature,Thermocouple,Voltage,Volume Flow RateRMS,anomaly
2020-01-01 00:00:00,0.412,0.398,0.221,0.541,0.718,0.336,0.902,0.104,0
2020-01-01 00:00:01,0.419,0.401,0.225,0.538,0.722,0.341,0.899,0.108,0`;

const MODEL_OPTIONS: ModelOption[] = [
  {
    id: "xgb",
    name: "XGBoost",
    shortName: "XGBoost",
    family: "Ensemble",
    recall: 0.8049,
    precision: 0.9944,
    f1: 0.8897,
  },
  {
    id: "rf",
    name: "Random Forest",
    shortName: "Random Forest",
    family: "Tree",
    recall: 0.8429,
    precision: 0.8635,
    f1: 0.853,
  },
  {
    id: "et",
    name: "Extra Trees",
    shortName: "Extra Trees",
    family: "Tree",
    recall: 0.7669,
    precision: 0.8989,
    f1: 0.8277,
  },
  {
    id: "gb",
    name: "Gradient Boosting",
    shortName: "Gradient Boost",
    family: "Boosting",
    recall: 0.8397,
    precision: 0.8811,
    f1: 0.8599,
  },
  {
    id: "lr",
    name: "Logistic Regression",
    shortName: "Logistic Reg.",
    family: "Linear",
    recall: 0.8455,
    precision: 0.9755,
    f1: 0.9058,
  },
  {
    id: "knn",
    name: "KNN",
    shortName: "KNN",
    family: "Instance",
    recall: 0.5544,
    precision: 0.8768,
    f1: 0.6793,
  },
  {
    id: "svm",
    name: "SVM",
    shortName: "SVM",
    family: "Kernel",
    recall: 0.369,
    precision: 0.9409,
    f1: 0.5301,
  },
  {
    id: "transformer",
    name: "TransformerFusionLite",
    shortName: "Transformer",
    family: "Deep Learning",
    recall: null,
    precision: null,
    f1: null,
    note: "Pending artifact",
  },
];

function normaliseHeader(value: string): string {
  return value.trim().replace(/^['"]|['"]$/g, "").toLowerCase();
}

function parseHeader(text: string): string[] {
  const firstLine = text.split(/\r?\n/).find((line) => line.trim().length > 0) ?? "";
  return firstLine.split(",").map((column) => column.trim().replace(/^['"]|['"]$/g, ""));
}

function validateCsvHeader(columns: string[]): FileValidation {
  const normalisedColumns = columns.map(normaliseHeader);
  const requiredColumns = [TIMESTAMP_COLUMN, ...SENSOR_COLUMNS];

  const missing = requiredColumns.filter(
    (column) => !normalisedColumns.includes(column.toLowerCase()),
  );

  const hasLabel = LABEL_COLUMNS.some((column) =>
    normalisedColumns.includes(column.toLowerCase()),
  );

  if (missing.length > 0) {
    return {
      state: "invalid",
      message: `Invalid schema — missing columns: ${missing.join(", ")}. Please upload a SKAB-format CSV.`,
    };
  }

  const labelNote = hasLabel
    ? "Label column detected."
    : "No label/anomaly column detected. Prediction can still run.";

  return {
    state: "valid",
    message: `Schema validated — ${SENSOR_COLUMNS.length} sensor columns detected. ${labelNote}`,
  };
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatMetric(value: number | null) {
  if (value === null) return "—";

  return (
    <>
      {(value * 100).toFixed(2)}
      <span className="unit">%</span>
    </>
  );
}

export default function Upload() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [validation, setValidation] = useState<FileValidation>({
    state: "idle",
    message: "",
  });
  const [selectedModelId, setSelectedModelId] = useState<ModelId>("xgb");
  const [threshold, setThreshold] = useState(0.5);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [apiNotice, setApiNotice] = useState("");

  const selectedModel = useMemo(
    () => MODEL_OPTIONS.find((model) => model.id === selectedModelId) ?? MODEL_OPTIONS[0],
    [selectedModelId],
  );

  const isValid = validation.state === "valid" && selectedFile !== null;

  const resetFile = () => {
    setSelectedFile(null);
    setValidation({
      state: "idle",
      message: "",
    });
    setApiNotice("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const readAndValidateFile = (file: File) => {
    setApiNotice("");

    if (!file.name.toLowerCase().endsWith(".csv")) {
      setSelectedFile(null);
      setValidation({
        state: "invalid",
        message: "Invalid file type — please upload a .csv file.",
      });
      return;
    }

    const reader = new FileReader();

    reader.onload = () => {
      const result = typeof reader.result === "string" ? reader.result : "";
      const columns = parseHeader(result);

      if (columns.length <= 1) {
        setSelectedFile(null);
        setValidation({
          state: "invalid",
          message: "Invalid CSV — could not detect a comma-separated header row.",
        });
        return;
      }

      const nextValidation = validateCsvHeader(columns);
      setSelectedFile(nextValidation.state === "valid" ? file : null);
      setValidation(nextValidation);
    };

    reader.onerror = () => {
      setSelectedFile(null);
      setValidation({
        state: "invalid",
        message: "Could not read the selected file. Please try again.",
      });
    };

    reader.readAsText(file.slice(0, 4096));
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) readAndValidateFile(file);
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragOver(false);

    const file = event.dataTransfer.files[0];
    if (file) readAndValidateFile(file);
  };

  const handleRunPrediction = async () => {
    if (!isValid || isRunning || !selectedFile) return;

    setIsRunning(true);
    setApiNotice("");

    try {
      await uploadCsv(selectedFile);
    } catch (error) {
      const message =
        error && typeof error === "object" && "message" in error
          ? String(error.message)
          : "Backend upload API is not available yet.";

      setApiNotice(
        `${message} Frontend validation passed, so you can still continue to the results preview while T1/T2 are being completed.`,
      );
    } finally {
      window.setTimeout(() => {
        setIsRunning(false);
        navigate("/results");
      }, 700);
    }
  };

  return (
    <>
      <Nav active="upload" />

      {isRunning && (
        <div className="spinner-overlay active" role="status" aria-live="polite">
          <div className="spinner-card">
            <div className="spinner" />
            <div className="spinner-msg">Running anomaly detection…</div>
          </div>
        </div>
      )}

      <section className="upload-section">
        <div className="breadcrumb">
          <Link className="breadcrumb-link" to="/">
            Home
          </Link>
          <span className="sep">/</span>
          <span className="current">New prediction</span>
        </div>

        <h2 className="upload-h">New prediction run</h2>
        <p className="upload-sub">
          Upload a SKAB-format CSV, select a trained model, adjust the anomaly
          threshold, and prepare the run for backend prediction.
        </p>

        <div className="sec-eyebrow">— step 1 · upload your data</div>

        <div
          className={`drop-zone ${isDragOver ? "drag-over" : ""} ${isValid ? "file-loaded" : ""}`}
          onClick={() => fileInputRef.current?.click()}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          role="button"
          tabIndex={0}
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
          <div className="drop-sub">
            or <span>browse files</span>
          </div>
          <div className="drop-hint">CSV · SKAB FORMAT · 8 SENSOR CHANNELS · UTF-8</div>
        </div>

        {validation.state === "invalid" && (
          <div className="upload-error visible">
            <span>{validation.message}</span>
          </div>
        )}

        {isValid && (
          <div className="upload-success visible">
            File uploaded successfully — ready to run prediction.
          </div>
        )}

        {selectedFile && (
          <div className="file-preview visible">
            <div className="fp-icon">CSV</div>

            <div>
              <div className="fp-name">{selectedFile.name}</div>
              <div className="fp-size">{formatFileSize(selectedFile.size)}</div>
            </div>

            <button className="fp-remove" onClick={resetFile} type="button">
              Remove ✕
            </button>
          </div>
        )}

        {apiNotice && (
          <div className="upload-warning visible">
            <span>{apiNotice}</span>
          </div>
        )}

        {isValid && <div className="validation-ok visible">{validation.message}</div>}

        <div className="schema-row">
          <span className="schema-label">REQUIRED COLUMNS</span>
          <span className="pill req">datetime</span>
          {SENSOR_COLUMNS.map((column) => (
            <span className="pill" key={column}>
              {column}
            </span>
          ))}
          <span className="pill optional">anomaly optional</span>
        </div>

        <div className="divider" />

        <div className="sec-eyebrow">— step 2 · select a model</div>

        <div className="model-grid">
          {MODEL_OPTIONS.map((model) => (
            <button
              className={`model-card ${selectedModelId === model.id ? "selected" : ""}`}
              key={model.id}
              onClick={() => setSelectedModelId(model.id)}
              type="button"
            >
              <div className="mc-tag">{model.family}</div>
              <div className="mc-name">{model.shortName}</div>
            </button>
          ))}
        </div>

        <div className="divider" />

        <div className="sec-eyebrow">— model performance summary</div>

        <div className="perf-panel">
          <div className="perf-panel-head">
            <div className="perf-panel-title">
              <span className="perf-panel-title-dot" />
              Test-set performance for selected model
            </div>
            <div className="perf-panel-active-model">{selectedModel.name}</div>
          </div>

          <div className="perf-metrics-grid">
            <div className="perf-metric">
              <div className="perf-metric-label">Recall</div>
              <div className="perf-metric-value">{formatMetric(selectedModel.recall)}</div>
            </div>

            <div className="perf-metric">
              <div className="perf-metric-label">Precision</div>
              <div className="perf-metric-value">{formatMetric(selectedModel.precision)}</div>
            </div>

            <div className="perf-metric">
              <div className="perf-metric-label">F1-Score</div>
              <div className="perf-metric-value">{formatMetric(selectedModel.f1)}</div>
            </div>
          </div>

          <div className="perf-panel-foot">
            <span className="perf-panel-foot-dot" />
            {selectedModel.note ??
              "Evaluated on SKAB held-out test set using the scaffold model registry."}
          </div>
        </div>

        <div className="divider" />

        <div className="sec-eyebrow">— step 3 · configure and run</div>

        <div className="threshold-panel">
          <div className="threshold-head">
            <div>
              <div className="threshold-title">Alert threshold</div>
              <div className="threshold-sub">
                Predicted probabilities at or above this value are treated as anomaly alerts.
              </div>
            </div>

            <div className="threshold-value">{threshold.toFixed(2)}</div>
          </div>

          <input
            aria-label="Alert threshold"
            className="threshold-slider"
            max="0.95"
            min="0.05"
            onChange={(event) => setThreshold(Number(event.target.value))}
            step="0.05"
            type="range"
            value={threshold}
          />
        </div>

        <button
          className="run-btn has-tooltip"
          data-tooltip={
            isValid ? "Run prediction with selected model" : "Upload a valid CSV file first to enable"
          }
          disabled={!isValid || isRunning}
          onClick={handleRunPrediction}
          type="button"
        >
          {isRunning ? "Running prediction…" : "Run prediction"}
        </button>

        <div className="divider" />

        <div className="sec-eyebrow">— expected file format</div>

        <div className="sample-aside sample-aside--inline">
          <div className="aside-h">SKAB CSV format</div>

          <p className="aside-sub">
            The backend scaffold expects one timestamp column, 8 numeric SKAB sensor
            columns, and an optional anomaly/label column for labelled evaluation.
          </p>

          <pre className="code-block">{SAMPLE_CSV}</pre>

          <a className="dl-btn" href={sampleCsvUrl()}>
            Download sample CSV
          </a>
        </div>
      </section>

      <Footer />
    </>
  );
}