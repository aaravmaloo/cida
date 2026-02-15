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
        self.decision_threshold = 0.5
        self.tokenizer = None
        self.session = None
        self.using_model = False

        self._load_calibration()
        self._load_model_runtime()

    def _load_calibration(self) -> None:
        calib_path = Path(self.settings.calibration_path)
        if not calib_path.exists():
            logger.warning("calibration_missing", path=str(calib_path))
            return

        try:
            payload = json.loads(calib_path.read_text(encoding="utf-8"))
            self.temperature = float(payload.get("temperature", 1.0))
            self.ece = float(payload.get("ece", 0.08))
            self.decision_threshold = float(payload.get("optimal_threshold", 0.5))
        except Exception:
            logger.exception("calibration_load_failed", path=str(calib_path))

    def _tokenizer_candidates(self, onnx_path: Path) -> list[Path | str]:
        candidates: list[Path | str] = []

        if self.settings.detector_tokenizer_path:
            candidates.append(Path(self.settings.detector_tokenizer_path))

        candidates.append(onnx_path.parent / "model")
        candidates.append(Path(self.settings.calibration_path).parent / "model")
        candidates.append(Path(self.settings.detector_onnx_path).parent / "runtime_bundle" / "model")
        candidates.append(self.settings.detector_model_name)
        return candidates

    def _load_tokenizer(self, onnx_path: Path) -> None:
        if AutoTokenizer is None:
            return

        for candidate in self._tokenizer_candidates(onnx_path):
            try:
                if isinstance(candidate, Path):
                    if not candidate.exists():
                        continue
                    self.tokenizer = AutoTokenizer.from_pretrained(candidate)
                    logger.info("tokenizer_loaded", source=str(candidate))
                    return

                self.tokenizer = AutoTokenizer.from_pretrained(
                    candidate,
                    local_files_only=not self.settings.detector_allow_remote_download,
                )
                logger.info("tokenizer_loaded", source=str(candidate))
                return
            except Exception:
                continue

        logger.warning("tokenizer_unavailable", candidates=[str(c) for c in self._tokenizer_candidates(onnx_path)])

    def _load_model_runtime(self) -> None:
        onnx_path = Path(self.settings.detector_onnx_path)

        if ort is not None and onnx_path.exists():
            try:
                self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                logger.info("onnx_session_loaded", path=str(onnx_path))
            except Exception:
                logger.exception("onnx_session_failed", path=str(onnx_path))
        else:
            logger.warning("onnx_missing_or_ort_unavailable", path=str(onnx_path), ort_available=bool(ort))

        self._load_tokenizer(onnx_path)
        self.using_model = self.session is not None and self.tokenizer is not None
        if not self.using_model:
            logger.warning("detector_using_heuristic_fallback")

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

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

    def _model_logit(self, text: str) -> float:
        if self.session is None or self.tokenizer is None:
            return self._heuristic_logit(text)

        encoded = self.tokenizer(
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

    def _confidence_band(self, prob: float) -> str:
        # Confidence is distance from tuned operating threshold, penalized by ECE.
        distance = abs(prob - self.decision_threshold)
        adjusted = max(0.0, distance - min(0.25, self.ece))
        if not self.using_model:
            adjusted = min(adjusted, 0.48)  # cap fallback confidence

        if adjusted > 0.30:
            return "high"
        if adjusted > 0.15:
            return "medium"
        return "low"

    def analyze(self, text: str) -> dict:
        normalized = normalize_text(text)
        metrics = compute_metrics(normalized)

        logit = self._model_logit(normalized)
        calibrated = logit / max(0.05, self.temperature)
        prob = self._sigmoid(calibrated)

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
            "model_version": self.settings.model_version,
        }


detector_service = DetectorService()
