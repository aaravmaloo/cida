from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../../train_data/balanced_ai_human_prompts.csv")
    parser.add_argument("--output-dir", default="../artifacts/latest")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    run_step([sys.executable, "-m", "src.train", "--csv", args.csv, "--output-dir", args.output_dir], root)
    run_step([sys.executable, "-m", "src.calibration", "--val-logits", f"{args.output_dir}/val_logits.npz", "--output", f"{args.output_dir}/calibration.json"], root)
    run_step([sys.executable, "-m", "src.export_onnx", "--model-dir", f"{args.output_dir}/model", "--output", f"{args.output_dir}/model.onnx"], root)
    run_step([sys.executable, "-m", "src.evaluate", "--test-logits", f"{args.output_dir}/test_logits.npz", "--calibration", f"{args.output_dir}/calibration.json", "--output", f"{args.output_dir}/eval_metrics.json"], root)
    run_step([sys.executable, "-m", "src.model_card"], root)


if __name__ == "__main__":
    main()
