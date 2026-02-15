export type ConfidenceBand = "low" | "medium" | "high";

export interface AnalyzeResponse {
  analysis_id: string;
  ai_probability: number;
  human_score: number;
  confidence_band: ConfidenceBand;
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
}

export interface HumanizeResponse {
  humanize_id: string;
  rewritten_text: string;
  diff_stats: {
    changed_tokens: number;
    change_ratio: number;
  };
  readability_delta: number;
  quality_flags: string[];
  latency_ms: number;
}

export interface ReportCreateResponse {
  report_id: string;
  status: "queued" | "processing" | "ready" | "failed";
}

