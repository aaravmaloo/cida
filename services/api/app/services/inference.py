from __future__ import annotations

import gc
import math
import re
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.text_metrics import compute_metrics
from app.utils.text import clamp, normalize_text

logger = get_logger(__name__)

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_LABEL_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


class DetectorService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client: httpx.Client | None = None
        self.inference_url = f"{self.settings.hf_inference_base_url}/{self.settings.hf_model}"
        self.using_model = False
        if not self.settings.hf_token:
            logger.warning("detector_hf_token_missing")
            return
        try:
            self.client = httpx.Client(
                timeout=httpx.Timeout(self.settings.hf_timeout_seconds),
                headers={
                    "Authorization": f"Bearer {self.settings.hf_token}",
                    "Accept": "application/json",
                },
            )
            logger.info("detector_hf_client_ready", model=self.settings.hf_model)
        except Exception:
            logger.exception("detector_hf_client_init_failed", model=self.settings.hf_model)
            self.client = None

    def release_model(self) -> None:
        # Detector inference is remote via Hugging Face.
        if self.client is not None:
            self.client.close()
            self.client = None
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
    def _normalize_label(value: str) -> str:
        return _LABEL_NORMALIZE_RE.sub("_", value.strip().lower()).strip("_")

    @staticmethod
    def _collect_label_scores(payload: Any) -> dict[str, float]:
        scores: dict[str, float] = {}
        stack: list[Any] = [payload]

        while stack:
            current = stack.pop()
            if isinstance(current, list):
                stack.extend(current)
                continue

            if not isinstance(current, dict):
                continue

            label = current.get("label")
            score = DetectorService._coerce_probability(current.get("score"))
            if isinstance(label, str) and score is not None:
                normalized_label = DetectorService._normalize_label(label)
                existing = scores.get(normalized_label, 0.0)
                scores[normalized_label] = max(existing, score)

            labels = current.get("labels")
            raw_scores = current.get("scores")
            if isinstance(labels, list) and isinstance(raw_scores, list):
                for raw_label, raw_score in zip(labels, raw_scores):
                    if not isinstance(raw_label, str):
                        continue
                    coerced = DetectorService._coerce_probability(raw_score)
                    if coerced is None:
                        continue
                    normalized_label = DetectorService._normalize_label(raw_label)
                    existing = scores.get(normalized_label, 0.0)
                    scores[normalized_label] = max(existing, coerced)

            for value in current.values():
                if isinstance(value, (list, dict)):
                    stack.append(value)

        return scores

    @staticmethod
    def _resolve_hf_probability(payload: Any) -> float | None:
        direct = DetectorService._coerce_probability(payload)
        if direct is not None:
            return direct

        if isinstance(payload, dict):
            for key in ("ai_probability", "probability", "score"):
                direct = DetectorService._coerce_probability(payload.get(key))
                if direct is not None:
                    return direct

        labels = DetectorService._collect_label_scores(payload)
        if not labels:
            return None

        negative_ai_tokens = ("not_ai", "non_ai", "no_ai")
        human_tokens = ("human", "real", "original", "authentic")

        for label, score in labels.items():
            if any(token in label for token in negative_ai_tokens):
                return float(clamp(1.0 - score, 0.0, 1.0))

        for label, score in labels.items():
            if any(token in label for token in human_tokens):
                return float(clamp(1.0 - score, 0.0, 1.0))

        ai_tokens = ("ai", "generated", "machine", "synthetic", "llm", "bot")
        for label, score in labels.items():
            if any(token in label for token in ai_tokens):
                return score

        if "label_1" in labels:
            return labels["label_1"]
        if "class_1" in labels:
            return labels["class_1"]

        if len(labels) == 1:
            return next(iter(labels.values()))

        if "label_0" in labels and len(labels) == 2:
            return float(clamp(1.0 - labels["label_0"], 0.0, 1.0))

        return None

    def _hf_probability(self, text: str) -> float | None:
        if self.client is None:
            return None

        truncated = text[: max(200, self.settings.hf_max_input_chars)]
        request_payload = {
            "inputs": truncated,
            "parameters": {
                "function_to_apply": "sigmoid",
                "top_k": 2,
            },
            "options": {"wait_for_model": True},
        }

        try:
            response = self.client.post(self.inference_url, json=request_payload)
        except Exception:
            logger.exception("detector_hf_inference_failed", model=self.settings.hf_model)
            return None

        if response.status_code >= 400:
            logger.warning(
                "detector_hf_inference_http_error",
                model=self.settings.hf_model,
                status_code=response.status_code,
                preview=response.text[:180],
            )
            return None

        try:
            payload = response.json()
        except ValueError:
            logger.warning("detector_hf_unparseable_response", preview=response.text[:180])
            return None

        if isinstance(payload, dict) and isinstance(payload.get("error"), str):
            logger.warning("detector_hf_inference_error", preview=payload["error"][:180])
            return None

        probability = self._resolve_hf_probability(payload)
        if probability is None:
            logger.warning("detector_hf_unparseable_response", preview=str(payload)[:180])
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
        probability = self._hf_probability(text)
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
            "model_version": self.settings.hf_model,
        }


detector_service = DetectorService()
