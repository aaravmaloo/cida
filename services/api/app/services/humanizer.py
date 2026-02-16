from __future__ import annotations

import difflib
import re
import time
from urllib.parse import quote

import httpx
from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.text_metrics import compute_metrics
from app.utils.text import normalize_text

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
        self._last_model_error = ""

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

    def _api_rewrite(self, text: str, preserve_terms: list[str]) -> str | None:
        if not self.settings.humanizer_use_remote_api:
            self._last_model_error = "remote_api_disabled"
            return None

        token = self.settings.hf_token.strip()
        if not token:
            self._last_model_error = "missing_hf_token"
            return None

        protected, mapping = self._protect_terms(text, preserve_terms)
        if self.settings.humanizer_api_url.strip():
            endpoint = self.settings.humanizer_api_url.strip()
        else:
            base_url = self.settings.hf_router_base_url.rstrip("/")
            encoded_model = quote(self.model_name, safe="")
            endpoint = f"{base_url}/hf-inference/models/{encoded_model}"
        payload = {
            "inputs": protected,
            "parameters": {"max_new_tokens": self.max_new_tokens, "return_full_text": False},
            "options": {"wait_for_model": True},
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=self.settings.humanizer_api_timeout_seconds) as client:
                response = client.post(endpoint, headers=headers, json=payload)
            if response.status_code >= 400:
                detail = response.text.strip().replace("\n", " ")
                if response.status_code == 404 and not self.settings.humanizer_api_url.strip():
                    self._last_model_error = (
                        "api_http_404:model_not_available_via_router; "
                        "set HUMANIZER_API_URL to a dedicated HF endpoint for this model"
                    )
                else:
                    self._last_model_error = f"api_http_{response.status_code}:{detail[:220]}"
                return None

            data = response.json()
            if isinstance(data, dict) and data.get("error"):
                error_detail = str(data["error"]).strip().replace("\n", " ")
                self._last_model_error = f"api_error:{error_detail[:220]}"
                return None

            rewritten = ""
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    for key in ("generated_text", "summary_text", "translation_text", "text"):
                        value = first.get(key)
                        if isinstance(value, str) and value.strip():
                            rewritten = value.strip()
                            break
            elif isinstance(data, dict):
                for key in ("generated_text", "summary_text", "translation_text", "text"):
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        rewritten = value.strip()
                        break

            if not rewritten:
                self._last_model_error = "api_unexpected_payload"
                return None

            rewritten = self._restore_terms(rewritten, mapping)
            rewritten = _SPACE_RE.sub(" ", rewritten).strip()
            self._last_model_error = ""
            logger.info("humanizer_api_success", model=self.model_name)
            return rewritten or None
        except Exception as exc:
            message = str(exc).strip().replace("\n", " ")
            self._last_model_error = f"api_request_failed:{type(exc).__name__}:{message}"
            logger.exception("humanizer_api_failed", model=self.model_name)
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
        mode = "huggingface_api"
        rewritten = self._api_rewrite(normalized, preserve_terms)
        if rewritten is None:
            if self.settings.humanizer_require_model:
                reason = self._last_model_error or "unknown"
                raise RuntimeError(f"Required humanizer model unavailable: {self.model_name} ({reason})")
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
