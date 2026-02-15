from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def nll_loss(temp: float, logits: np.ndarray, labels: np.ndarray) -> float:
    scaled = logits / max(0.05, temp)
    probs = np.clip(sigmoid(scaled), 1e-6, 1 - 1e-6)
    labels = labels.astype(np.float64)
    return float(-(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)).mean())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> tuple[float, list[dict]]:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    breakdown = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < bins - 1 else probs <= hi)
        if mask.sum() == 0:
            breakdown.append({"bin": i, "count": 0, "acc": 0.0, "conf": 0.0, "gap": 0.0})
            continue
        acc = float(labels[mask].mean())
        conf = float(probs[mask].mean())
        gap = abs(acc - conf)
        ece += gap * (mask.sum() / len(probs))
        breakdown.append({"bin": i, "count": int(mask.sum()), "acc": acc, "conf": conf, "gap": float(gap)})
    return float(ece), breakdown


def best_threshold(probs: np.ndarray, labels: np.ndarray) -> dict:
    best = {"threshold": 0.5, "f1": -1.0, "accuracy": -1.0}
    for threshold in np.linspace(0.05, 0.95, 181):
        preds = (probs >= threshold).astype(np.int64)
        f1 = float(f1_score(labels, preds, zero_division=0))
        acc = float(accuracy_score(labels, preds))
        if f1 > best["f1"] or (abs(f1 - best["f1"]) <= 1e-9 and acc > best["accuracy"]):
            best = {"threshold": float(threshold), "f1": f1, "accuracy": acc}
    return best


def run(args: argparse.Namespace) -> None:
    data = np.load(args.val_logits)
    logits = data["logits"].astype(np.float64)
    labels = data["labels"].astype(np.int64)

    result = minimize(lambda x: nll_loss(float(x[0]), logits, labels), x0=[1.0], bounds=[(0.05, 10.0)])
    temperature = float(result.x[0])

    probs = sigmoid(logits / temperature)
    ece, bins = expected_calibration_error(probs, labels, bins=args.ece_bins)
    threshold = best_threshold(probs, labels)

    payload = {
        "temperature": temperature,
        "ece": ece,
        "optimal_threshold": float(threshold["threshold"]),
        "threshold_metrics": threshold,
        "reliability_bins": bins,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-logits", default="../artifacts/latest/val_logits.npz")
    parser.add_argument("--output", default="../artifacts/latest/calibration.json")
    parser.add_argument("--ece-bins", type=int, default=15)
    run(parser.parse_args())
