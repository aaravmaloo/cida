from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def nll_loss(temp: float, logits: np.ndarray, labels: np.ndarray) -> float:
    scaled = logits / max(0.05, temp)
    probs = np.clip(sigmoid(scaled), 1e-6, 1 - 1e-6)
    labels = labels.astype(np.float64)
    return float(-(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)).mean())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    breakdown = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < bins - 1 else probs <= hi)
        if mask.sum() == 0:
            breakdown.append({"bin": i, "count": 0, "acc": 0.0, "conf": 0.0})
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        gap = abs(acc - conf)
        ece += gap * (mask.sum() / len(probs))
        breakdown.append({"bin": i, "count": int(mask.sum()), "acc": float(acc), "conf": float(conf)})
    return float(ece), breakdown


def run(args: argparse.Namespace) -> None:
    data = np.load(args.val_logits)
    logits = data["logits"].astype(np.float64)
    labels = data["labels"].astype(np.float64)

    result = minimize(lambda x: nll_loss(float(x[0]), logits, labels), x0=[1.0], bounds=[(0.05, 10.0)])
    temperature = float(result.x[0])

    probs = sigmoid(logits / temperature)
    ece, bins = expected_calibration_error(probs, labels, bins=10)

    payload = {
        "temperature": temperature,
        "ece": ece,
        "reliability_bins": bins,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-logits", default="../artifacts/latest/val_logits.npz")
    parser.add_argument("--output", default="../artifacts/latest/calibration.json")
    run(parser.parse_args())

