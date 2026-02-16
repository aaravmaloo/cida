from __future__ import annotations

import difflib
import re
import time

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.text_metrics import compute_metrics
from app.utils.text import normalize_text

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

logger = get_logger(__name__)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SPACE_RE = re.compile(r"\s+")
_HUMANEYES_MODEL_ID = "Eemansleepdeprived/Humaneyes"


class HumanizerService:
    def __init__(self, max_input_tokens: int | None = None) -> None:
        self.settings = get_settings()
        self.model_name = _HUMANEYES_MODEL_ID
        if self.settings.humanizer_model_name and self.settings.humanizer_model_name != _HUMANEYES_MODEL_ID:
            logger.warning(
                "humanizer_model_forced_to_humaneyes",
                configured=self.settings.humanizer_model_name,
                forced=_HUMANEYES_MODEL_ID,
            )
        self.max_input_tokens = max_input_tokens or self.settings.humanizer_max_input_tokens
        self.max_new_tokens = self.settings.humanizer_max_new_tokens
        self._tokenizer = None
        self._model = None
        self._model_unavailable = False
        self._device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    def _get_model(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        if self._model_unavailable:
            return None
        if torch is None or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            self._model_unavailable = True
            logger.warning("humanizer_runtime_unavailable", reason="missing_torch_or_transformers")
            return None
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=not self.settings.humanizer_allow_remote_download,
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                local_files_only=not self.settings.humanizer_allow_remote_download,
            )
            self._model.to(self._device)
            self._model.eval()
            logger.info("humanizer_model_loaded", model=self.model_name, device=self._device)
            return self._tokenizer, self._model
        except Exception:
            self._model_unavailable = True
            self._tokenizer = None
            self._model = None
            logger.exception("humanizer_model_load_failed", model=self.model_name)
            return None

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

    def _model_rewrite(self, text: str, preserve_terms: list[str]) -> str | None:
        loaded = self._get_model()
        if loaded is None or torch is None:
            return None

        tokenizer, model = loaded
        protected, mapping = self._protect_terms(text, preserve_terms)
        try:
            encoded = tokenizer(
                protected,
                max_length=self.max_input_tokens,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            max_new_tokens = min(self.max_new_tokens, max(64, int(encoded["input_ids"].shape[-1] * 1.4)))
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            rewritten = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            rewritten = self._restore_terms(rewritten, mapping)
            rewritten = _SPACE_RE.sub(" ", rewritten).strip()
            return rewritten or None
        except Exception:
            logger.exception("humanizer_inference_failed", model=self.model_name)
            return None

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
        mode = "huggingface"
        rewritten = self._model_rewrite(normalized, preserve_terms)
        if rewritten is None:
            if self.settings.humanizer_require_model:
                raise RuntimeError(f"Required humanizer model unavailable: {self.model_name}")
            mode = "fallback"
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
        if mode == "fallback":
            quality_flags.append("fallback_mode")

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
            "humanizer_mode": mode,
            "humanizer_model": self.model_name,
        }


humanizer_service = HumanizerService()
