from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def safe_roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, probs))


def metrics_at_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "roc_auc": safe_roc_auc(labels, probs),
        "brier": float(brier_score_loss(labels, probs)),
    }


def run(args: argparse.Namespace) -> None:
    test_data = np.load(args.test_logits)
    logits = test_data["logits"].astype(np.float64)
    labels = test_data["labels"].astype(np.int64)

    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    temperature = float(calibration.get("temperature", 1.0))
    threshold = float(calibration.get("optimal_threshold", 0.5))

    probs = sigmoid(logits / max(0.05, temperature))

    payload = {
        "temperature": temperature,
        "optimal_threshold": threshold,
        "metrics@0.5": metrics_at_threshold(probs, labels, threshold=0.5),
        "metrics@optimal_threshold": metrics_at_threshold(probs, labels, threshold=threshold),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-logits", default="../artifacts/latest/test_logits.npz")
    parser.add_argument("--calibration", default="../artifacts/latest/calibration.json")
    parser.add_argument("--output", default="../artifacts/latest/eval_metrics.json")
    run(parser.parse_args())
