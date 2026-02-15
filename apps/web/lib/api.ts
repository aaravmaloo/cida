export type AnalyzeResponse = {
  analysis_id: string;
  ai_probability: number;
  human_score: number;
  confidence_band: "low" | "medium" | "high";
  readability: {
    flesch_reading_ease: number;
    flesch_kincaid_grade: number;
    smog_index: number;
  };
  complexity: number;
  burstiness: number;
  vocab_diversity: number;
  word_count: number;
  estimated_read_time: number;
  model_version: string;
  latency_ms: number;
};

export type HumanizeResponse = {
  humanize_id: string;
  rewritten_text: string;
  diff_stats: {
    changed_tokens: number;
    change_ratio: number;
  };
  readability_delta: number;
  quality_flags: string[];
  latency_ms: number;
};

export type ReportCreateResponse = {
  report_id: string;
  status: "queued" | "processing" | "ready" | "failed";
};

export type ReportStatusResponse = {
  report_id: string;
  status: "queued" | "processing" | "ready" | "failed";
  json_url?: string;
  pdf_url?: string;
  error_message?: string;
};

export type AnalyticsSummary = {
  total_analyses: number;
  total_humanizations: number;
  avg_ai_probability: number;
  p95_latency_ms: number;
  confidence_distribution: Record<string, number>;
  abuse_block_count: number;
};

const DEFAULT_API_BASE = "http://localhost:8000";

function resolveApiBase(raw: string | undefined): string {
  const value = (raw ?? DEFAULT_API_BASE).trim();
  if (!value) return DEFAULT_API_BASE;

  const withProtocol = /^https?:\/\//i.test(value) ? value : `https://${value}`;

  try {
    const url = new URL(withProtocol);
    if (url.hostname.endsWith(".railway.internal")) {
      console.warn(
        "NEXT_PUBLIC_API_BASE_URL points to a Railway internal hostname. Use your API public Railway domain (e.g. https://<service>.up.railway.app).",
      );
    }
    return url.toString().replace(/\/$/, "");
  } catch {
    console.warn(`Invalid NEXT_PUBLIC_API_BASE_URL: "${value}". Falling back to ${DEFAULT_API_BASE}.`);
    return DEFAULT_API_BASE;
  }
}

const API_BASE = resolveApiBase(process.env.NEXT_PUBLIC_API_BASE_URL);

async function handle<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail =
      typeof payload?.detail === "string"
        ? payload.detail
        : typeof payload?.message === "string"
          ? payload.message
          : "Request failed";
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }
  return (await response.json()) as T;
}

export async function analyzeText(text: string): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_BASE}/v1/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, source: "paste" }),
  });
  return handle<AnalyzeResponse>(response);
}

export async function analyzeFile(file: File): Promise<AnalyzeResponse> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("source", "upload");

  const response = await fetch(`${API_BASE}/v1/analyze`, {
    method: "POST",
    body: fd,
  });
  return handle<AnalyzeResponse>(response);
}

export async function humanizeText(payload: {
  text: string;
  style: "natural" | "formal" | "concise";
  strength: number;
  preserve_terms: string[];
}): Promise<HumanizeResponse> {
  const response = await fetch(`${API_BASE}/v1/humanize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handle<HumanizeResponse>(response);
}

export async function createReport(analysisId: string): Promise<ReportCreateResponse> {
  const response = await fetch(`${API_BASE}/v1/reports`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis_id: analysisId }),
  });
  return handle<ReportCreateResponse>(response);
}

export async function getReportStatus(reportId: string): Promise<ReportStatusResponse> {
  const response = await fetch(`${API_BASE}/v1/reports/${reportId}`);
  return handle<ReportStatusResponse>(response);
}

export async function adminLogin(passkey: string): Promise<{ access_token: string }> {
  const response = await fetch(`${API_BASE}/v1/admin/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ passkey }),
  });
  return handle<{ access_token: string }>(response);
}

export async function fetchAnalytics(token: string): Promise<AnalyticsSummary> {
  const response = await fetch(`${API_BASE}/v1/analytics/summary`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  return handle<AnalyticsSummary>(response);
}

