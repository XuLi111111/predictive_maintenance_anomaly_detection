/**
 * Single REST client for the pump.detect backend.
 * The streaming WebSocket lives in `streamClient.ts`.
 */

export const API_BASE = "/api";

// ─── Types mirrored from backend ────────────────────────────────────

export interface ApiModel {
  id: string;
  name: string;
  family: string;
  artifact: string;
  is_dl: boolean;
  recall: number | null;
  precision: number | null;
  f1: number | null;
  requires_scaling: boolean;
  unavailable: boolean;
}

export interface UploadResponse {
  file_id: string;
  rows: number;
  time_range: { start: string; end: string };
  columns_detected: string[];
  has_label: boolean;
  warnings: string[];
  model_default: string;
}

export interface AnomalyWindow {
  start_idx: number;
  end_idx: number;
  peak_prob: number;
}

export interface PredictResult {
  model_id: string;
  probs: number[];
  anomaly_windows: AnomalyWindow[];
  peak_idx: number;
  peak_prob: number;
  total_windows: number;
  fault_windows: number;
  metrics: { precision: number; recall: number; f1: number } | null;
  unavailable: boolean;
  error: string | null;
}

export interface CompareResponse {
  models: PredictResult[];
}

export interface ApiError {
  status: number;
  message: string;
}

// ─── Helpers ────────────────────────────────────────────────────────

async function parseError(response: Response): Promise<ApiError> {
  try {
    const payload = await response.json();
    const detail = payload?.detail;
    return {
      status: response.status,
      message: typeof detail === "string" ? detail : response.statusText,
    };
  } catch {
    return {
      status: response.status,
      message: response.statusText || "Request failed",
    };
  }
}

// ─── Endpoints ──────────────────────────────────────────────────────

export function sampleCsvUrl(): string {
  return `${API_BASE}/sample-csv`;
}

export async function listModels(): Promise<ApiModel[]> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<ApiModel[]>;
}

export async function uploadCsv(file: File): Promise<UploadResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: fd });
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<UploadResponse>;
}

export async function runPredict(
  fileId: string,
  modelId: string,
  threshold: number,
): Promise<PredictResult> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_id: fileId, model_id: modelId, threshold }),
  });
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<PredictResult>;
}

export async function runCompare(
  fileId: string,
  threshold: number,
): Promise<CompareResponse> {
  const res = await fetch(`${API_BASE}/predict/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_id: fileId, threshold }),
  });
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<CompareResponse>;
}

/** POST /api/report and trigger a browser download. */
export async function downloadReport(
  fileId: string,
  modelId: string,
): Promise<void> {
  const res = await fetch(`${API_BASE}/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_id: fileId, model_id: modelId }),
  });
  if (!res.ok) throw await parseError(res);

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `pump_detect_report_${fileId.slice(0, 8)}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
