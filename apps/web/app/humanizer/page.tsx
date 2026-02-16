"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation } from "@tanstack/react-query";
import Link from "next/link";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { humanizeText } from "@/lib/api";

const schema = z.object({
  text: z.string().min(40),
  style: z.enum(["natural", "formal", "concise"]),
  strength: z.coerce.number().int().min(1).max(3),
  preserve_terms: z.string().optional(),
});

type FormData = z.infer<typeof schema>;

export default function HumanizerPage() {
  const form = useForm<FormData>({
    resolver: zodResolver(schema),
    defaultValues: {
      text: "",
      style: "natural",
      strength: 2,
      preserve_terms: "",
    },
  });

  const mutation = useMutation({
    mutationFn: (data: FormData) =>
      humanizeText({
        text: data.text,
        style: data.style,
        strength: data.strength,
        preserve_terms: data.preserve_terms
          ? data.preserve_terms
              .split(",")
              .map((x) => x.trim())
              .filter(Boolean)
          : [],
      }),
  });

  return (
    <main className="mx-auto my-8 max-w-5xl rounded-2xl border border-cida-line bg-cida-panel p-8 shadow-panel">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="font-heading text-3xl font-semibold">Humanizer</h1>
        <Link href="/" className="rounded-lg border border-cida-line px-3 py-2 text-sm hover:bg-white">
          Back to Detector
        </Link>
      </div>

      <form
        className="space-y-4"
        onSubmit={form.handleSubmit((data) => {
          mutation.mutate(data);
        })}
      >
        <textarea
          aria-label="Humanizer input"
          {...form.register("text")}
          className="h-56 w-full resize-none rounded-xl border border-cida-line bg-white p-4"
          placeholder="Paste text to improve clarity and natural flow."
        />

        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <label className="text-sm">
            Style
            <select {...form.register("style")} className="mt-1 w-full rounded-lg border border-cida-line bg-white p-2">
              <option value="natural">Natural</option>
              <option value="formal">Formal</option>
              <option value="concise">Concise</option>
            </select>
          </label>

          <label className="text-sm">
            Strength (1-3)
            <input
              type="number"
              {...form.register("strength")}
              min={1}
              max={3}
              className="mt-1 w-full rounded-lg border border-cida-line bg-white p-2"
            />
          </label>

          <label className="text-sm">
            Preserve terms (comma-separated)
            <input
              {...form.register("preserve_terms")}
              className="mt-1 w-full rounded-lg border border-cida-line bg-white p-2"
              placeholder="API, SQL, DeBERTa"
            />
          </label>
        </div>

        <button
          type="submit"
          className="rounded-xl bg-cida-accent px-6 py-3 font-semibold text-white disabled:opacity-60"
          disabled={mutation.isPending}
        >
          {mutation.isPending ? "Humanizing..." : "Humanize Text"}
        </button>
      </form>

      {mutation.error && <p className="mt-4 text-sm text-red-600">{(mutation.error as Error).message}</p>}

      {mutation.data && (
        <section className="mt-6 rounded-xl border border-cida-line bg-white p-4">
          <h2 className="font-heading text-xl font-semibold">Output</h2>
          <p className="mt-2 whitespace-pre-wrap text-sm">{mutation.data.rewritten_text}</p>
          <div className="mt-4 grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
            <p>
              Changed tokens: <strong>{mutation.data.diff_stats.changed_tokens}</strong>
            </p>
            <p>
              Change ratio: <strong>{Math.round(mutation.data.diff_stats.change_ratio * 100)}%</strong>
            </p>
            <p>
              Readability delta: <strong>{mutation.data.readability_delta}</strong>
            </p>
            <p>
              Latency: <strong>{mutation.data.latency_ms} ms</strong>
            </p>
          </div>
          <div className="mt-3 text-xs text-cida-mute">
            <p>
              Runtime: <strong>{mutation.data.humanizer_mode}</strong>
            </p>
            <p>
              Model: <strong>{mutation.data.humanizer_model}</strong>
            </p>
          </div>
        </section>
      )}
    </main>
  );
}

