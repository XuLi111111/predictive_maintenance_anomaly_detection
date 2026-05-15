const BASE = "/api";

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
  metrics: Record<string, number> | null;
}

export async function runPredict(
  fileId: string,
  modelId: string,
  threshold: number
): Promise<PredictResult> {
  const res = await fetch(`${BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_id: fileId, model_id: modelId, threshold }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<PredictResult>;
}

export async function runCompare(
  fileId: string,
  threshold: number
): Promise<PredictResult[]> {
  const res = await fetch(`${BASE}/predict/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_id: fileId, threshold }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<PredictResult[]>;
export const API_BASE = "/api";

export type ApiModel = {
  id: string;
  name: string;
  family: string;
  artifact?: string;
  is_dl?: boolean;
  recall: number;
  precision: number;
  f1: number;
};

export type UploadResponse = {
  file_id: string;
  rows: number;
  time_range?: {
    start?: string;
    end?: string;
  };
  columns_detected: string[];
  has_label: boolean;
  warnings: string[];
};

export type ApiError = {
  status: number;
  message: string;
};

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

export function sampleCsvUrl(): string {
  return `${API_BASE}/sample-csv`;
}

export async function listModels(): Promise<ApiModel[]> {
  const response = await fetch(`${API_BASE}/models`);

  if (!response.ok) {
    throw await parseError(response);
  }

  return response.json() as Promise<ApiModel[]>;
}

export async function uploadCsv(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw await parseError(response);
  }

  return response.json() as Promise<UploadResponse>;
}