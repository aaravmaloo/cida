from __future__ import annotations

import difflib
import re
import time

from app.services.text_metrics import compute_metrics
from app.utils.text import normalize_text

try:
    from transformers import pipeline
except Exception:  # pragma: no cover
    pipeline = None


class HumanizerService:
    def __init__(self, max_input_tokens: int = 1800) -> None:
        self.max_input_tokens = max_input_tokens
        self._pipeline = None
        self._pipeline_unavailable = False

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if self._pipeline_unavailable:
            return None
        if pipeline is None:
            self._pipeline_unavailable = True
            return None
        try:
            self._pipeline = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                tokenizer="google/flan-t5-base",
            )
        except Exception:
            self._pipeline_unavailable = True
            self._pipeline = None
        return self._pipeline

    @staticmethod
    def _fallback_rewrite(text: str, style: str, strength: int) -> str:
        replacements = {
            "utilize": "use",
            "in order to": "to",
            "therefore": "so",
            "however": "but",
            "moreover": "also",
        }
        out = text
        for src, dst in replacements.items():
            out = re.sub(rf"\b{re.escape(src)}\b", dst, out, flags=re.IGNORECASE)
        if style == "concise":
            out = re.sub(r"\s+", " ", out)
        if strength >= 3:
            out = out.replace(" very ", " ")
        return out.strip()

    def rewrite(self, *, text: str, style: str, strength: int, preserve_terms: list[str]) -> dict:
        start = time.perf_counter()
        normalized = normalize_text(text)
        tokens = normalized.split()
        if len(tokens) > self.max_input_tokens:
            normalized = " ".join(tokens[: self.max_input_tokens])

        input_metrics = compute_metrics(normalized)

        prompt = (
            f"Rewrite the following text for {style} clarity with natural flow, "
            f"preserving meaning and terms: {', '.join(preserve_terms) if preserve_terms else 'none'}. "
            f"Rewrite strength {strength}/3.\n\nText:\n{normalized}"
        )

        pipe = self._get_pipeline()
        if pipe is not None:
            try:
                result = pipe(prompt, max_new_tokens=max(128, int(len(tokens) * 1.25)), do_sample=False)
                rewritten = result[0]["generated_text"].strip()
            except Exception:
                rewritten = self._fallback_rewrite(normalized, style, strength)
        else:
            rewritten = self._fallback_rewrite(normalized, style, strength)

        output_metrics = compute_metrics(rewritten)

        matcher = difflib.SequenceMatcher(None, normalized.split(), rewritten.split())
        ratio = matcher.ratio()
        changed_tokens = int((1 - ratio) * max(len(normalized.split()), len(rewritten.split())))

        latency_ms = (time.perf_counter() - start) * 1000

        quality_flags = []
        if abs(output_metrics.readability["flesch_kincaid_grade"] - input_metrics.readability["flesch_kincaid_grade"]) > 4.5:
            quality_flags.append("readability_shift")
        if len(rewritten.split()) < max(1, len(normalized.split()) // 3):
            quality_flags.append("over_compression")

        return {
            "rewritten_text": rewritten,
            "diff_stats": {
                "changed_tokens": changed_tokens,
                "change_ratio": round(1 - ratio, 4),
            },
            "readability_delta": round(
                output_metrics.readability["flesch_kincaid_grade"] - input_metrics.readability["flesch_kincaid_grade"],
                3,
            ),
            "quality_flags": quality_flags,
            "latency_ms": round(latency_ms, 3),
            "input_word_count": input_metrics.word_count,
            "output_word_count": output_metrics.word_count,
        }


humanizer_service = HumanizerService()

