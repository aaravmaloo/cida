from __future__ import annotations

import gc
import json
import math
import re
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.text_metrics import compute_metrics
from app.utils.text import clamp, normalize_text

logger = get_logger(__name__)

try:
    from groq import Groq
except Exception:  # pragma: no cover
    Groq = None


_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_PROBABILITY_RE = re.compile(r"ai[_\s-]*probability[^0-9-]*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class DetectorService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = None
        self.using_model = False
        if Groq is None:
            logger.warning("detector_groq_sdk_unavailable")
            return
        if not self.settings.groq_api_key:
            logger.warning("detector_groq_api_key_missing")
            return
        try:
            self.client = Groq(api_key=self.settings.groq_api_key)
            logger.info("detector_groq_client_ready", model=self.settings.groq_model)
        except Exception:
            logger.exception("detector_groq_client_init_failed", model=self.settings.groq_model)
            self.client = None

    def release_model(self) -> None:
        # Detector inference is remote via Groq; nothing to unload locally.
        gc.collect()
        logger.info("detector_model_release_noop")

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _coerce_probability(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            numeric = float(value)
        elif isinstance(value, str):
            match = _NUMBER_RE.search(value.strip())
            if not match:
                return None
            numeric = float(match.group(0))
        else:
            return None

        if numeric > 1.0 and numeric <= 100.0:
            numeric /= 100.0
        return float(clamp(numeric, 0.0, 1.0))

    @staticmethod
    def _extract_probability(raw: str) -> float | None:
        candidate = (raw or "").strip()
        if not candidate:
            return None

        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\s*```$", "", candidate)

        objects: list[str] = [candidate]
        match_obj = _OBJECT_RE.search(candidate)
        if match_obj:
            objects.append(match_obj.group(0))

        for obj in objects:
            try:
                payload = json.loads(obj)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                for key in ("ai_probability", "probability", "score"):
                    value = payload.get(key)
                    coerced = DetectorService._coerce_probability(value)
                    if coerced is not None:
                        return coerced

        match_prob = _PROBABILITY_RE.search(candidate)
        if match_prob:
            return DetectorService._coerce_probability(match_prob.group(1))

        return DetectorService._coerce_probability(candidate)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                else:
                    text = getattr(item, "text", None) or getattr(item, "content", None)
                if isinstance(text, str):
                    parts.append(text)
            return "".join(parts)
        return str(content or "")

    def _groq_probability(self, text: str) -> float | None:
        if self.client is None:
            return None

        truncated = text[: max(200, self.settings.groq_max_input_chars)]
        system_prompt = (
            "You score whether text is AI-generated. "
            "Return only JSON: {\"ai_probability\": <float between 0 and 1>}."
        )
        user_prompt = f"Analyze this text:\n\n{truncated}"
        request_kwargs = {
            "model": self.settings.groq_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.settings.groq_temperature,
            "max_completion_tokens": self.settings.groq_max_completion_tokens,
            "top_p": self.settings.groq_top_p,
            "reasoning_effort": self.settings.groq_reasoning_effort,
            "stream": False,
        }

        try:
            completion = self.client.chat.completions.create(**request_kwargs)
        except TypeError:
            request_kwargs.pop("reasoning_effort", None)
            try:
                completion = self.client.chat.completions.create(**request_kwargs)
            except Exception:
                logger.exception("detector_groq_inference_failed", model=self.settings.groq_model)
                return None
        except Exception:
            logger.exception("detector_groq_inference_failed", model=self.settings.groq_model)
            return None

        content = ""
        if getattr(completion, "choices", None):
            first_choice = completion.choices[0]
            message = getattr(first_choice, "message", None)
            content = self._content_to_text(getattr(message, "content", ""))

        probability = self._extract_probability(content)
        if probability is None:
            logger.warning("detector_groq_unparseable_response", preview=content[:180])
        return probability

    def _heuristic_logit(self, text: str) -> float:
        m = compute_metrics(text)
        avg_word_len = (sum(len(w) for w in text.split()) / max(1, m.word_count)) if m.word_count else 0.0

        # Heuristic fallback calibrated around typical long-form prose stats.
        logit = (
            (m.complexity - 0.58) * 2.4
            - (m.vocab_diversity - 0.60) * 1.2
            - (m.burstiness - 0.24) * 1.8
            + ((m.readability["flesch_kincaid_grade"] - 9.0) / 8.0) * 0.9
            + ((avg_word_len - 4.8) / 2.0) * 0.35
        )
        return float(clamp(logit, -6.0, 6.0))

    def _model_probability(self, text: str) -> float:
        probability = self._groq_probability(text)
        if probability is not None:
            self.using_model = True
            return probability

        self.using_model = False
        return self._sigmoid(self._heuristic_logit(text))

    def _confidence_band(self, prob: float) -> str:
        distance = abs(prob - 0.5)
        adjusted = distance
        if not self.using_model:
            adjusted = min(adjusted, 0.28)

        if adjusted > 0.30:
            return "high"
        if adjusted > 0.15:
            return "medium"
        return "low"

    def analyze(self, text: str) -> dict:
        normalized = normalize_text(text)
        metrics = compute_metrics(normalized)

        prob = self._model_probability(normalized)

        return {
            "ai_probability": round(prob, 6),
            "human_score": round(1.0 - prob, 6),
            "confidence_band": self._confidence_band(prob),
            "readability": metrics.readability,
            "complexity": metrics.complexity,
            "burstiness": metrics.burstiness,
            "vocab_diversity": metrics.vocab_diversity,
            "word_count": metrics.word_count,
            "estimated_read_time": metrics.estimated_read_time,
            "model_version": self.settings.groq_model,
        }


detector_service = DetectorService()
