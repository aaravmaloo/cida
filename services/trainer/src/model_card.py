from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_model_card(artifact_dir: str) -> None:
    root = Path(artifact_dir)
    metrics = _load_json(root / "metrics.json")
    evaluation = _load_json(root / "eval_metrics.json")
    calibration = _load_json(root / "calibration.json")
    training_config = _load_json(root / "training_config.json")
    export_summary = _load_json(root / "export_summary.json")

    payload = {
        "model": {
            "base_model": training_config.get("model_name", "unknown"),
            "task": "binary ai/human content detection",
            "labels": {"0": "human", "1": "ai"},
        },
        "dataset": {
            "source": training_config.get("data_files")
            or training_config.get("csv", "../../train_data/balanced_ai_human_prompts.csv"),
            "text_column": training_config.get("text_col", "text"),
            "label_column": training_config.get("label_col", "label"),
        },
        "artifacts": {
            "onnx": str(root / "model.onnx"),
            "runtime_bundle": str(root / "runtime_bundle"),
            "calibration": str(root / "calibration.json"),
            "metrics": str(root / "metrics.json"),
            "evaluation": str(root / "eval_metrics.json"),
            "training_config": str(root / "training_config.json"),
        },
        "training": training_config,
        "results": {
            "metrics": metrics,
            "evaluation": evaluation,
            "calibration": {
                "temperature": calibration.get("temperature"),
                "ece": calibration.get("ece"),
                "optimal_threshold": calibration.get("optimal_threshold"),
            },
            "export": export_summary,
        },
        "runtime_notes": [
            "Use model.onnx + runtime_bundle/model tokenizer files together.",
            "Use calibration.json optimal_threshold for production decisioning.",
            "If model artifacts are unavailable, API falls back to heuristic scoring.",
        ],
    }

    (root / "model_card.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="../artifacts/latest")
    args = parser.parse_args()
    write_model_card(args.artifact_dir)
