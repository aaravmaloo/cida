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


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SPACE_RE = re.compile(r"\s+")


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
    def _protect_terms(text: str, preserve_terms: list[str]) -> tuple[str, dict[str, str]]:
        protected = text
        mapping: dict[str, str] = {}
        for idx, term in enumerate([t.strip() for t in preserve_terms if t.strip()]):
            key = f"__KEEP_TERM_{idx}__"
            mapping[key] = term
            protected = re.sub(re.escape(term), key, protected, flags=re.IGNORECASE)
        return protected, mapping

    @staticmethod
    def _restore_terms(text: str, mapping: dict[str, str]) -> str:
        out = text
        for key, term in mapping.items():
            out = out.replace(key, term)
        return out

    @staticmethod
    def _split_long_sentence(sentence: str) -> list[str]:
        parts = [p.strip(" ,") for p in sentence.split(",") if p.strip()]
        if len(parts) <= 1:
            return [sentence.strip()]
        out = []
        for part in parts:
            chunk = part[0].upper() + part[1:] if part and part[0].islower() else part
            if not chunk.endswith((".", "!", "?")):
                chunk += "."
            out.append(chunk)
        return out

    @staticmethod
    def _fallback_rewrite(text: str, style: str, strength: int, preserve_terms: list[str]) -> str:
        protected, mapping = HumanizerService._protect_terms(text, preserve_terms)
        out = _SPACE_RE.sub(" ", protected).strip()

        natural_replacements = {
            "utilize": "use",
            "in order to": "to",
            "therefore": "so",
            "however": "but",
            "moreover": "also",
            "furthermore": "also",
            "it is important to note that": "",
            "it can be observed that": "",
        }
        concise_replacements = {
            "it should be noted that": "",
            "due to the fact that": "because",
            "a large number of": "many",
            "in the event that": "if",
            "at this point in time": "now",
        }
        formal_replacements = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "isn't": "is not",
            "it's": "it is",
        }

        if style in {"natural", "concise"}:
            for src, dst in natural_replacements.items():
                out = re.sub(rf"\b{re.escape(src)}\b", dst, out, flags=re.IGNORECASE)
        if style == "concise":
            for src, dst in concise_replacements.items():
                out = re.sub(rf"\b{re.escape(src)}\b", dst, out, flags=re.IGNORECASE)
            out = re.sub(r"\b(really|very|extremely|highly)\b", "", out, flags=re.IGNORECASE)
        if style == "formal":
            for src, dst in formal_replacements.items():
                out = re.sub(rf"\b{re.escape(src)}\b", dst, out, flags=re.IGNORECASE)

        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(out) if s.strip()]
        rewritten_sentences: list[str] = []

        for sentence in sentences:
            words = sentence.split()
            if style == "concise" and (len(words) > 30 or (strength >= 2 and len(words) > 22)):
                rewritten_sentences.extend(HumanizerService._split_long_sentence(sentence))
            elif style == "natural" and strength >= 3 and len(words) > 34:
                rewritten_sentences.extend(HumanizerService._split_long_sentence(sentence))
            else:
                rewritten_sentences.append(sentence if sentence.endswith((".", "!", "?")) else f"{sentence}.")

        out = " ".join(rewritten_sentences)
        out = re.sub(r"\s+([,.!?;:])", r"\1", out)
        out = _SPACE_RE.sub(" ", out).strip()
        out = HumanizerService._restore_terms(out, mapping)

        return out

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
                rewritten = self._fallback_rewrite(normalized, style, strength, preserve_terms)
        else:
            rewritten = self._fallback_rewrite(normalized, style, strength, preserve_terms)

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
        if ratio > 0.95:
            quality_flags.append("minimal_change")

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
