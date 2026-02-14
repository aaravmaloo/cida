from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    n = y_true.shape[0]
    n_pos = int((y_true == 1).sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-indexed average rank
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_pos = float(ranks[y_true == 1].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = y_true.astype(np.int64).reshape(-1)
    y_prob = y_prob.astype(np.float64).reshape(-1)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = max(1, len(y_true))

    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    auc = _roc_auc_binary(y_true, y_prob)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    y_true = y_true.astype(np.int64).reshape(-1)
    y_prob = y_prob.astype(np.float64).reshape(-1)
    thresholds = np.unique(y_prob)
    if thresholds.size == 0:
        return 0.5, binary_metrics(y_true, y_prob, threshold=0.5)


    if thresholds.size > 512:
        idx = np.linspace(0, thresholds.size - 1, num=512, dtype=np.int64)
        thresholds = thresholds[idx]

    best_thr = 0.5
    best = binary_metrics(y_true, y_prob, threshold=best_thr)
    for thr in thresholds:
        m = binary_metrics(y_true, y_prob, threshold=float(thr))
        if m["f1"] > best["f1"]:
            best = m
            best_thr = float(thr)
    return best_thr, best


def save_metrics(metrics: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")

