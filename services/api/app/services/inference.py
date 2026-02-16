from __future__ import annotations

import math

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.text_metrics import compute_metrics
from app.utils.text import clamp, normalize_text

logger = get_logger(__name__)

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


class DetectorService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self.using_model = False

        self._load_model_runtime()

    def _resolve_ai_label_index(self, num_labels: int) -> int:
        if self.model is None or num_labels <= 1:
            return 0

        config = getattr(self.model, "config", None)
        if config is not None:
            label2id = getattr(config, "label2id", None) or {}
            for label, idx in label2id.items():
                if "ai" in str(label).lower():
                    candidate = int(idx)
                    if 0 <= candidate < num_labels:
                        return candidate

            id2label = getattr(config, "id2label", None) or {}
            for idx, label in id2label.items():
                if "ai" in str(label).lower():
                    candidate = int(idx)
                    if 0 <= candidate < num_labels:
                        return candidate

        configured = int(clamp(self.settings.detector_ai_label, 0, max(0, num_labels - 1)))
        return configured

    def _load_model_runtime(self) -> None:
        if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            logger.warning("detector_runtime_unavailable", reason="missing_torch_or_transformers")
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.settings.detector_model_name,
                local_files_only=not self.settings.detector_allow_remote_download,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.settings.detector_model_name,
                local_files_only=not self.settings.detector_allow_remote_download,
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            self.using_model = True
            logger.info("detector_model_loaded", model=self.settings.detector_model_name, device=self.device)
        except Exception:
            logger.exception("detector_model_load_failed", model=self.settings.detector_model_name)
            self.tokenizer = None
            self.model = None
            self.using_model = False

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

    def _model_probability(self, text: str) -> float:
        if self.model is None or self.tokenizer is None or torch is None:
            return self._sigmoid(self._heuristic_logit(text))

        try:
            encoded = self.tokenizer(
                text,
                max_length=self.settings.detector_max_length,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits

            if logits.ndim == 1 or logits.shape[-1] == 1:
                prob = float(torch.sigmoid(logits.reshape(-1)[0]).item())
                return float(clamp(prob, 0.0, 1.0))

            probs = torch.softmax(logits, dim=-1)[0]
            ai_index = self._resolve_ai_label_index(probs.shape[-1])
            prob = float(probs[ai_index].item())
            return float(clamp(prob, 0.0, 1.0))
        except Exception:
            logger.exception("detector_inference_failed", model=self.settings.detector_model_name)
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
            "model_version": self.settings.model_version,
        }


detector_service = DetectorService()
