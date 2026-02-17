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

_NEGATIVE_AI_TOKENS = ("not_ai", "non_ai", "no_ai")
_HUMAN_TOKENS = ("human", "real", "original", "authentic")
_AI_TOKENS = ("ai", "generated", "machine", "synthetic", "llm", "bot")


class DetectorService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client: httpx.Client | None = None
        self.using_model = False

        headers = {"Accept": "application/json"}
        if self.settings.hf_space_api_token:
            headers["Authorization"] = f"Bearer {self.settings.hf_space_api_token}"

        try:
            self.client = httpx.Client(
                timeout=httpx.Timeout(self.settings.hf_space_timeout_seconds),
                headers=headers,
            )
            logger.info(
                "detector_hf_space_client_ready",
                model=self.settings.hf_space_model_version,
                endpoint=self.settings.hf_space_predict_url,
            )
        except Exception:
            logger.exception(
                "detector_hf_space_client_init_failed",
                endpoint=self.settings.hf_space_predict_url,
            )
            self.client = None

    def release_model(self) -> None:
        # Detector inference is remote via HF Space.
        if self.client is not None:
            self.client.close()
            self.client = None
        gc.collect()
        logger.info("detector_model_release_noop")

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _normalize_label(value: str) -> str:
        return _LABEL_NORMALIZE_RE.sub("_", value.strip().lower()).strip("_")

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
    def _collect_label_scores(payload: Any) -> dict[str, float]:
        scores: dict[str, float] = {}
        stack: list[Any] = [payload]

        def set_score(label: str, raw_score: Any) -> None:
            coerced = DetectorService._coerce_probability(raw_score)
            if coerced is None:
                return
            normalized_label = DetectorService._normalize_label(label)
            if not normalized_label:
                return
            existing = scores.get(normalized_label, 0.0)
            scores[normalized_label] = max(existing, coerced)

        while stack:
            current = stack.pop()

            if isinstance(current, list):
                stack.extend(current)
                if len(current) == 2 and isinstance(current[0], str):
                    set_score(current[0], current[1])
                continue

            if not isinstance(current, dict):
                continue

            label = current.get("label")
            if isinstance(label, str):
                for score_key in ("score", "confidence", "probability"):
                    if score_key in current:
                        set_score(label, current.get(score_key))

            labels = current.get("labels")
            raw_scores = current.get("scores")
            if isinstance(labels, list) and isinstance(raw_scores, list):
                for raw_label, raw_score in zip(labels, raw_scores):
                    if isinstance(raw_label, str):
                        set_score(raw_label, raw_score)

            if isinstance(raw_scores, dict):
                for raw_label, raw_score in raw_scores.items():
                    if isinstance(raw_label, str):
                        set_score(raw_label, raw_score)

            for key in ("ai", "human", "generated", "machine", "not_ai"):
                if key in current:
                    set_score(key, current.get(key))

            for value in current.values():
                if isinstance(value, (list, dict)):
                    stack.append(value)

        return scores

    @staticmethod
    def _extract_predicted_label(payload: Any) -> str | None:
        candidates: list[str] = []
        stack: list[Any] = [payload]

        while stack:
            current = stack.pop()
            if isinstance(current, str):
                normalized = DetectorService._normalize_label(current)
                if normalized:
                    candidates.append(normalized)
                continue

            if isinstance(current, list):
                stack.extend(current)
                continue

            if not isinstance(current, dict):
                continue

            for key in ("predicted_label", "prediction", "result", "output_label", "label"):
                value = current.get(key)
                if isinstance(value, str):
                    normalized = DetectorService._normalize_label(value)
                    if normalized:
                        candidates.append(normalized)

            for value in current.values():
                if isinstance(value, (list, dict, str)):
                    stack.append(value)

        for label in candidates:
            if any(token in label for token in _NEGATIVE_AI_TOKENS):
                return label
            if any(token in label for token in _HUMAN_TOKENS):
                return label
            if any(token in label for token in _AI_TOKENS):
                return label
        return None

    @staticmethod
    def _resolve_space_probability(payload: Any) -> float | None:
        if isinstance(payload, dict):
            for key in ("ai_probability", "probability", "score"):
                direct = DetectorService._coerce_probability(payload.get(key))
                if direct is not None:
                    return direct

        labels = DetectorService._collect_label_scores(payload)
        if labels:
            for label, score in labels.items():
                if any(token in label for token in _NEGATIVE_AI_TOKENS):
                    return float(clamp(1.0 - score, 0.0, 1.0))

            for label, score in labels.items():
                if any(token in label for token in _HUMAN_TOKENS):
                    return float(clamp(1.0 - score, 0.0, 1.0))

            for label, score in labels.items():
                if any(token in label for token in _AI_TOKENS):
                    return score

            if "label_1" in labels:
                return labels["label_1"]
            if "class_1" in labels:
                return labels["class_1"]
            if "label_0" in labels and len(labels) == 2:
                return float(clamp(1.0 - labels["label_0"], 0.0, 1.0))
            if len(labels) == 1:
                return next(iter(labels.values()))

        predicted_label = DetectorService._extract_predicted_label(payload)
        if predicted_label is not None:
            if any(token in predicted_label for token in _NEGATIVE_AI_TOKENS):
                return 0.0
            if any(token in predicted_label for token in _HUMAN_TOKENS):
                return 0.0
            if any(token in predicted_label for token in _AI_TOKENS):
                return 1.0

        return None

    def _space_probability(self, text: str) -> float | None:
        if self.client is None:
            return None

        truncated = text[: max(200, self.settings.hf_space_max_input_chars)]
        request_payload = {"data": [truncated]}

        try:
            response = self.client.post(self.settings.hf_space_predict_url, json=request_payload)
        except Exception:
            logger.exception(
                "detector_hf_space_inference_failed",
                endpoint=self.settings.hf_space_predict_url,
            )
            return None

        if response.status_code >= 400:
            logger.warning(
                "detector_hf_space_http_error",
                endpoint=self.settings.hf_space_predict_url,
                status_code=response.status_code,
                preview=response.text[:180],
            )
            return None

        try:
            payload = response.json()
        except ValueError:
            logger.warning("detector_hf_space_unparseable_response", preview=response.text[:180])
            return None

        if isinstance(payload, dict) and isinstance(payload.get("error"), str):
            logger.warning("detector_hf_space_inference_error", preview=payload["error"][:180])
            return None

        probability = self._resolve_space_probability(payload)
        if probability is None:
            logger.warning("detector_hf_space_unparseable_response", preview=str(payload)[:180])
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
        probability = self._space_probability(text)
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
            "model_version": self.settings.hf_space_model_version,
        }


detector_service = DetectorService()
