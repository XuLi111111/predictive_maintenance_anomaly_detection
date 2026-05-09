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
}