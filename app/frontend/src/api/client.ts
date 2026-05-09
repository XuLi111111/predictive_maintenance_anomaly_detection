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