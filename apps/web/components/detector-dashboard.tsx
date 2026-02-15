"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation } from "@tanstack/react-query";
import { Pie, PieChart, Cell, ResponsiveContainer } from "recharts";
import { ChangeEvent, useMemo, useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";

import {
  AnalyzeResponse,
  analyzeFile,
  analyzeText,
  createReport,
  getReportStatus,
} from "@/lib/api";

const schema = z.object({
  text: z.string().min(60, "Minimum 60 words required."),
});

type FormData = z.infer<typeof schema>;

const FALLBACK_RESULT: AnalyzeResponse = {
  analysis_id: "-",
  ai_probability: 0.75,
  human_score: 0.25,
  confidence_band: "high",
  readability: {
    flesch_reading_ease: 56,
    flesch_kincaid_grade: 12.1,
    smog_index: 12.6,
  },
  complexity: 0.88,
  burstiness: 0.15,
  vocab_diversity: 0.42,
  word_count: 0,
  estimated_read_time: 0,
  model_version: "deberta-v3-base-v1",
  latency_ms: 0,
};

function percent(v: number): string {
  return `${Math.round(v * 100)}%`;
}

export function DetectorDashboard() {
  const [result, setResult] = useState<AnalyzeResponse>(FALLBACK_RESULT);
  const [error, setError] = useState<string>("");
  const [reportStatus, setReportStatus] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const form = useForm<FormData>({
    resolver: zodResolver(schema),
    defaultValues: { text: "" },
  });

  const textValue = form.watch("text") || "";
  const wordCount = useMemo(() => textValue.trim().split(/\s+/).filter(Boolean).length, [textValue]);
  const readTime = useMemo(() => (wordCount / 230).toFixed(1), [wordCount]);

  const analyzeMutation = useMutation({
    mutationFn: (payload: { text?: string; file?: File }) => {
      if (payload.file) {
        return analyzeFile(payload.file);
      }
      return analyzeText(payload.text || "");
    },
    onSuccess: (data) => {
      setError("");
      setResult(data);
    },
    onError: (err: Error) => {
      setError(err.message);
    },
  });

  const handleSubmit = form.handleSubmit(async (data) => {
    await analyzeMutation.mutateAsync({ text: data.text });
  });

  const handleImportClick = () => fileInputRef.current?.click();

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    await analyzeMutation.mutateAsync({ file });
  };

  const handleDownloadReport = async () => {
    if (!result.analysis_id || result.analysis_id === "-") return;
    setReportStatus("Queueing report...");

    const create = await createReport(result.analysis_id);
    let retries = 20;
    while (retries > 0) {
      await new Promise((r) => setTimeout(r, 1500));
      const status = await getReportStatus(create.report_id);
      setReportStatus(`Report ${status.status}...`);
      if (status.status === "ready") {
        if (status.pdf_url) {
          window.open(status.pdf_url, "_blank", "noopener,noreferrer");
        }
        setReportStatus("Report ready");
        return;
      }
      if (status.status === "failed") {
        setReportStatus(status.error_message || "Report failed");
        return;
      }
      retries -= 1;
    }
    setReportStatus("Report timeout. Try again.");
  };

  const pieData = [
    { name: "AI", value: Math.round(result.ai_probability * 100) },
    { name: "Human", value: 100 - Math.round(result.ai_probability * 100) },
  ];

  return (
    <section>
      <div className="grid min-h-[760px] grid-cols-1 lg:grid-cols-[1fr_340px]">
        <section className="border-b border-cida-line p-8 lg:border-b-0 lg:border-r">
          <h1 className="font-heading text-4xl font-semibold">AI Content Analysis</h1>
          <p className="mt-2 text-cida-mute">Paste text or upload TXT, DOCX, or PDF for AI likelihood and linguistic analysis.</p>

          <form onSubmit={handleSubmit} className="mt-6 rounded-2xl border border-cida-line bg-white">
            <div className="flex items-center justify-between border-b border-cida-line px-4 py-3 text-sm text-cida-mute">
              <div className="flex items-center gap-4">
                <button type="button" onClick={handleImportClick} className="font-semibold text-cida-accent hover:underline">
                  Import
                </button>
                <span>Paste</span>
              </div>
              <span className="text-xs">MIN. 60 WORDS REQUIRED</span>
            </div>

            <textarea
              aria-label="Content input"
              {...form.register("text")}
              className="h-[420px] w-full resize-none border-0 px-5 py-4 text-lg text-cida-text outline-none"
              placeholder="Start typing or paste your content here for deep neural analysis..."
            />

            <div className="flex items-center justify-between border-t border-cida-line px-5 py-4">
              <div className="flex gap-8 text-sm text-cida-mute">
                <div>
                  <p className="text-xs uppercase">Word count</p>
                  <p className="font-semibold text-cida-text">{wordCount} words</p>
                </div>
                <div>
                  <p className="text-xs uppercase">Est. read time</p>
                  <p className="font-semibold text-cida-text">{readTime} min</p>
                </div>
              </div>
              <button
                type="submit"
                className="rounded-xl bg-cida-accent px-8 py-3 font-semibold text-white shadow-md hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={analyzeMutation.isPending}
              >
                {analyzeMutation.isPending ? "Scanning..." : "Scan Text"}
              </button>
            </div>
          </form>

          <input
            aria-label="Import file"
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".txt,.pdf,.docx"
            onChange={handleFileChange}
          />

          {form.formState.errors.text && <p className="mt-3 text-sm text-red-600">{form.formState.errors.text.message}</p>}
          {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
        </section>

        <aside className="space-y-4 p-6">
          <div className="rounded-2xl border border-cida-line bg-white p-5">
            <p className="text-center text-xs font-semibold uppercase tracking-wider text-cida-mute">AI Likelihood</p>
            <div className="mx-auto mt-3 h-44 w-44">
              <ResponsiveContainer>
                <PieChart>
                  <Pie data={pieData} dataKey="value" innerRadius={58} outerRadius={74} stroke="none">
                    <Cell fill="#1d5fff" />
                    <Cell fill="#e9edf7" />
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <p className="-mt-28 text-center text-5xl font-bold">{percent(result.ai_probability)}</p>
            <p className="mt-20 text-center text-sm font-semibold text-cida-accent">
              {result.ai_probability >= 0.7 ? "Highly Likely AI" : result.ai_probability >= 0.45 ? "Mixed Signal" : "Likely Human"}
            </p>
            <p className="mt-4 text-center text-sm text-cida-mute">
              Score combines calibrated transformer probability and linguistic pattern analysis.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-2xl border border-cida-line bg-white p-4">
              <p className="text-xs uppercase text-cida-mute">Human score</p>
              <p className="mt-1 text-3xl font-bold">{percent(result.human_score)}</p>
            </div>
            <div className="rounded-2xl border border-cida-line bg-white p-4">
              <p className="text-xs uppercase text-cida-mute">Readability</p>
              <p className="mt-1 text-3xl font-bold">G{Math.round(result.readability.flesch_kincaid_grade)}</p>
            </div>
          </div>

          <div className="rounded-2xl border border-cida-line bg-white p-4">
            <p className="mb-4 text-xs font-semibold uppercase text-cida-mute">Content breakdown</p>

            <div className="space-y-3">
              {[
                { label: "Sentence Complexity", value: result.complexity },
                { label: "Vocabulary Variety", value: result.vocab_diversity },
                { label: "Burstiness Index", value: result.burstiness },
              ].map((item) => (
                <div key={item.label}>
                  <div className="mb-1 flex justify-between text-sm">
                    <span>{item.label}</span>
                    <span className="font-semibold text-cida-accent">{percent(item.value)}</span>
                  </div>
                  <div className="h-2 rounded-full bg-[#e7ecf7]">
                    <div className="h-2 rounded-full bg-cida-accent" style={{ width: percent(item.value) }} />
                  </div>
                </div>
              ))}
            </div>

            <button
              type="button"
              className="mt-6 w-full rounded-xl border border-cida-line bg-cida-panel px-4 py-3 font-semibold text-cida-accent hover:border-cida-accent"
              onClick={handleDownloadReport}
            >
              Download Full Report
            </button>
            {reportStatus && <p className="mt-2 text-center text-xs text-cida-mute">{reportStatus}</p>}
          </div>
        </aside>
      </div>

      <footer className="flex flex-wrap items-center justify-between border-t border-cida-line px-8 py-4 text-sm text-cida-mute">
        <div className="flex gap-5">
          <a href="#">API Documentation</a>
          <a href="#">Pricing Plans</a>
          <a href="#">About Cida</a>
          <a href="#">Security</a>
        </div>
        <p>Ã‚Â© 2026 Cida AI Analytics.</p>
      </footer>
    </section>
  );
}

