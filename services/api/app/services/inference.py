from __future__ import annotations

import gc
import json
import math
import re
from typing import Any
from urllib.parse import urlsplit

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
        self.predict_url = self.settings.hf_space_predict_url
        self._discovery_attempted = False

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

    def _discover_queue_predict_url(self) -> str | None:
        if self.client is None:
            return None
        if self._discovery_attempted:
            return None

        self._discovery_attempted = True
        parsed = urlsplit(self.settings.hf_space_predict_url)
        if not parsed.scheme or not parsed.netloc:
            return None
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        try:
            response = self.client.get(f"{base_url}/gradio_api/info")
        except Exception:
            logger.exception("detector_hf_space_info_fetch_failed", base_url=base_url)
            return None

        if response.status_code >= 400:
            logger.warning(
                "detector_hf_space_info_http_error",
                base_url=base_url,
                status_code=response.status_code,
                preview=response.text[:180],
            )
            return None

        try:
            payload = response.json()
        except ValueError:
            logger.warning("detector_hf_space_info_unparseable_response", preview=response.text[:180])
            return None

        named_endpoints = payload.get("named_endpoints") if isinstance(payload, dict) else None
        if not isinstance(named_endpoints, dict) or not named_endpoints:
            return None

        preferred = ("/detect_ai_content", "/predict")
        endpoint_name = next((name for name in preferred if name in named_endpoints), None)
        if endpoint_name is None:
            endpoint_name = next(iter(named_endpoints.keys()), None)
        if not isinstance(endpoint_name, str) or not endpoint_name.strip():
            return None

        clean_name = endpoint_name.strip().lstrip("/")
        if not clean_name:
            return None

        discovered = f"{base_url}/gradio_api/call/{clean_name}"
        logger.info("detector_hf_space_endpoint_discovered", endpoint=discovered)
        return discovered

    def _parse_queue_sse_payload(self, raw_text: str) -> Any | None:
        data_lines = [line[len("data:") :].strip() for line in raw_text.splitlines() if line.startswith("data:")]
        for item in reversed(data_lines):
            if not item:
                continue
            try:
                return json.loads(item)
            except json.JSONDecodeError:
                continue
        return None

    def _call_space_endpoint(self, endpoint_url: str, request_payload: dict[str, Any]) -> Any | None:
        if self.client is None:
            return None

        try:
            response = self.client.post(endpoint_url, json=request_payload)
        except Exception:
            logger.exception("detector_hf_space_inference_failed", endpoint=endpoint_url)
            return None

        if response.status_code >= 400:
            logger.warning(
                "detector_hf_space_http_error",
                endpoint=endpoint_url,
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

        if isinstance(payload, dict) and isinstance(payload.get("event_id"), str):
            event_id = payload["event_id"].strip()
            if not event_id:
                return None
            stream_url = f"{endpoint_url.rstrip('/')}/{event_id}"
            try:
                stream_response = self.client.get(stream_url)
            except Exception:
                logger.exception("detector_hf_space_queue_fetch_failed", endpoint=stream_url)
                return None

            if stream_response.status_code >= 400:
                logger.warning(
                    "detector_hf_space_queue_http_error",
                    endpoint=stream_url,
                    status_code=stream_response.status_code,
                    preview=stream_response.text[:180],
                )
                return None

            queue_payload = self._parse_queue_sse_payload(stream_response.text)
            if queue_payload is None:
                logger.warning("detector_hf_space_queue_unparseable_response", preview=stream_response.text[:180])
                return None
            return queue_payload

        return payload

    def _extract_label_scores(self, result: Any) -> dict[str, float]:
        scores: dict[str, float] = {}

        def set_score(label: str, raw_score: Any) -> None:
            normalized = self._normalize_label(label)
            if not normalized:
                return
            coerced = self._coerce_probability(raw_score)
            if coerced is None:
                return
            current = scores.get(normalized, 0.0)
            scores[normalized] = max(current, coerced)

        if isinstance(result, dict):
            for key in ("AI-written", "Human-written", "ai", "human"):
                if key in result:
                    set_score(key, result.get(key))

            confidences = result.get("confidences")
            if isinstance(confidences, list):
                for item in confidences:
                    if isinstance(item, dict):
                        label = item.get("label")
                        confidence = item.get("confidence")
                        if isinstance(label, str):
                            set_score(label, confidence)
            elif isinstance(confidences, dict):
                for label, confidence in confidences.items():
                    if isinstance(label, str):
                        set_score(label, confidence)

            labels = result.get("labels")
            raw_scores = result.get("scores")
            if isinstance(labels, list) and isinstance(raw_scores, list):
                for label, score in zip(labels, raw_scores):
                    if isinstance(label, str):
                        set_score(label, score)

        return scores

    def _extract_data_payload(self, payload: Any) -> list[Any]:
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload, list):
            return payload
        return [payload]

    def _resolve_space_probability(self, payload: Any) -> float | None:
        data_payload = self._extract_data_payload(payload)
        first_item = data_payload[0] if data_payload else payload
        label_scores = self._extract_label_scores(first_item)

        ai_probability: float | None = None
        human_probability: float | None = None

        for label, score in label_scores.items():
            if any(token in label for token in _NEGATIVE_AI_TOKENS):
                human_probability = max(human_probability or 0.0, score)
            elif any(token in label for token in _HUMAN_TOKENS):
                human_probability = max(human_probability or 0.0, score)
            elif any(token in label for token in _AI_TOKENS):
                ai_probability = max(ai_probability or 0.0, score)

        if ai_probability is not None:
            return ai_probability
        if human_probability is not None:
            return float(clamp(1.0 - human_probability, 0.0, 1.0))

        predicted_label: str | None = None
        if len(data_payload) > 1 and isinstance(data_payload[1], str):
            predicted_label = self._normalize_label(data_payload[1])
        elif isinstance(first_item, dict) and isinstance(first_item.get("label"), str):
            predicted_label = self._normalize_label(first_item["label"])

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

        payload = self._call_space_endpoint(self.predict_url, request_payload)
        if payload is None:
            discovered = self._discover_queue_predict_url()
            if discovered and discovered != self.predict_url:
                fallback_payload = self._call_space_endpoint(discovered, request_payload)
                if fallback_payload is not None:
                    self.predict_url = discovered
                    payload = fallback_payload
                    logger.info("detector_hf_space_endpoint_switched", endpoint=discovered)

        if payload is None:
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
