from __future__ import annotations

import json
from pathlib import Path


def write_model_card(artifact_dir: str) -> None:
    root = Path(artifact_dir)
    metrics_path = root / "metrics.json"
    eval_path = root / "eval_metrics.json"
    calib_path = root / "calibration.json"

    payload = {
        "model": "microsoft/deberta-v3-base",
        "task": "binary ai/human content detection",
        "dataset": "train_data/balanced_ai_human_prompts.csv",
        "artifacts": {
            "onnx": str(root / "model.onnx"),
            "calibration": str(calib_path),
            "metrics": str(metrics_path),
            "evaluation": str(eval_path),
        },
        "labels": {"0": "human", "1": "ai"},
    }

    (root / "model_card.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    write_model_card("../artifacts/latest")

