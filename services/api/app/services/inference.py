from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.text_metrics import compute_metrics
from app.utils.text import clamp, normalize_text

logger = get_logger(__name__)

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    ort = None
    AutoTokenizer = None


class DetectorService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.temperature = 1.0
        self.ece = 0.08
        self.tokenizer = None
        self._tokenizer_load_attempted = False
        self.session = None

        calib_path = Path(self.settings.calibration_path)
        if calib_path.exists():
            payload = json.loads(calib_path.read_text(encoding="utf-8"))
            self.temperature = float(payload.get("temperature", 1.0))
            self.ece = float(payload.get("ece", 0.08))

        onnx_path = Path(self.settings.detector_onnx_path)
        if ort is not None and onnx_path.exists():
            providers = ["CPUExecutionProvider"]
            try:
                self.session = ort.InferenceSession(str(onnx_path), providers=providers)
            except Exception:
                logger.warning("onnx_session_failed", path=str(onnx_path))

    def _load_tokenizer_once(self):
        if self.tokenizer is not None or self._tokenizer_load_attempted:
            return self.tokenizer

        self._tokenizer_load_attempted = True
        if AutoTokenizer is None:
            return None

        try:
            # Avoid startup/runtime hangs on platforms without model cache access.
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.settings.detector_model_name,
                local_files_only=True,
            )
        except Exception:
            logger.warning("tokenizer_load_failed", model=self.settings.detector_model_name)
            self.tokenizer = None
        return self.tokenizer

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _heuristic_logit(self, text: str) -> float:
        m = compute_metrics(text)
        avg_word_len = (sum(len(w) for w in text.split()) / max(1, m.word_count)) if m.word_count else 0.0
        logit = (
            (m.complexity - 0.55) * 2.1
            - (m.vocab_diversity - 0.58) * 1.1
            - (m.burstiness - 0.25) * 1.6
            + ((m.readability["flesch_kincaid_grade"] - 9.5) / 8.0) * 0.8
            + ((avg_word_len - 4.8) / 2.5) * 0.4
        )
        return float(clamp(logit, -6.0, 6.0))

    def _model_logit(self, text: str) -> float:
        tokenizer = self._load_tokenizer_once()
        if self.session is None or tokenizer is None:
            return self._heuristic_logit(text)

        encoded = tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="np",
            padding="max_length",
        )
        feeds = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        outputs = self.session.run(None, feeds)
        logits = outputs[0]
        return float(logits.reshape(-1)[0])

    def analyze(self, text: str) -> dict:
        normalized = normalize_text(text)
        metrics = compute_metrics(normalized)

        logit = self._model_logit(normalized)
        calibrated = logit / max(0.05, self.temperature)
        prob = self._sigmoid(calibrated)

        confidence = abs(prob - 0.5) * 2
        reliability_penalty = min(0.25, self.ece)
        adjusted = max(0.0, confidence - reliability_penalty)
        if adjusted > 0.66:
            band = "high"
        elif adjusted > 0.33:
            band = "medium"
        else:
            band = "low"

        return {
            "ai_probability": round(prob, 6),
            "human_score": round(1.0 - prob, 6),
            "confidence_band": band,
            "readability": metrics.readability,
            "complexity": metrics.complexity,
            "burstiness": metrics.burstiness,
            "vocab_diversity": metrics.vocab_diversity,
            "word_count": metrics.word_count,
            "estimated_read_time": metrics.estimated_read_time,
            "model_version": self.settings.model_version,
        }


detector_service = DetectorService()

