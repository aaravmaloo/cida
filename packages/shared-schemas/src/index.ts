export type ConfidenceBand = "low" | "medium" | "high";

export interface AnalyzeResponse {
  analysis_id: string;
  ai_probability: number;
  human_score: number;
  predicted_label: "AI" | "Human";
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

export interface ReportCreateResponse {
  report_id: string;
  status: "queued" | "processing" | "ready" | "failed";
}

