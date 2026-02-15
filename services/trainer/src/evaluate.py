from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def run(args: argparse.Namespace) -> None:
    logits = np.load(args.test_logits)["logits"].astype(np.float64)
    labels = np.load(args.test_logits)["labels"].astype(np.int64)

    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    temp = float(calibration.get("temperature", 1.0))

    probs = sigmoid(logits / max(0.05, temp))
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "ece": float(calibration.get("ece", 0.0)),
    }

    Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-logits", default="../artifacts/latest/test_logits.npz")
    parser.add_argument("--calibration", default="../artifacts/latest/calibration.json")
    parser.add_argument("--output", default="../artifacts/latest/eval_metrics.json")
    run(parser.parse_args())

