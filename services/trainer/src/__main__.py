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
    parser = argparse.ArgumentParser(description="End-to-end detector training pipeline")
    parser.add_argument("--data-path", dest="data_paths", action="append", help="Training file path (.csv/.parquet). Repeat to include multiple files.")
    parser.add_argument("--csv", default="../../train_data/balanced_ai_human_prompts.csv")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="auto")
    parser.add_argument("--unlabeled-default-label", type=int, choices=[0, 1], default=None)
    parser.add_argument("--output-dir", default="../artifacts/latest")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-large")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--attention-heads", type=int, default=16)
    parser.add_argument("--ffn-size", type=int, default=4096)
    parser.add_argument("--max-position-embeddings", type=int, default=514)
    parser.add_argument("--min-params", type=int, default=100_000_000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--tpu-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    train_cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--output-dir",
        args.output_dir,
        "--model-name",
        args.model_name,
        "--text-col",
        args.text_col,
        "--label-col",
        args.label_col,
        "--layers",
        str(args.layers),
        "--hidden-size",
        str(args.hidden_size),
        "--attention-heads",
        str(args.attention_heads),
        "--ffn-size",
        str(args.ffn_size),
        "--max-position-embeddings",
        str(args.max_position_embeddings),
        "--min-params",
        str(args.min_params),
        "--epochs",
        str(args.epochs),
    ]
    train_cmd.append("--tpu-bf16" if args.tpu_bf16 else "--no-tpu-bf16")
    if args.from_scratch:
        train_cmd.append("--from-scratch")
    if args.unlabeled_default_label is not None:
        train_cmd.extend(["--unlabeled-default-label", str(args.unlabeled_default_label)])
    if args.data_paths:
        for data_path in args.data_paths:
            train_cmd.extend(["--data-path", data_path])
    else:
        train_cmd.extend(["--csv", args.csv])
    run_step(train_cmd, root)
    run_step(
        [
            sys.executable,
            "-m",
            "src.calibration",
            "--val-logits",
            f"{args.output_dir}/val_logits.npz",
            "--output",
            f"{args.output_dir}/calibration.json",
        ],
        root,
    )
    run_step(
        [
            sys.executable,
            "-m",
            "src.evaluate",
            "--test-logits",
            f"{args.output_dir}/test_logits.npz",
            "--calibration",
            f"{args.output_dir}/calibration.json",
            "--output",
            f"{args.output_dir}/eval_metrics.json",
        ],
        root,
    )
    export_cmd = [
        sys.executable,
        "-m",
        "src.export_onnx",
        "--model-dir",
        f"{args.output_dir}/model",
        "--output",
        f"{args.output_dir}/model.onnx",
        "--bundle-dir",
        f"{args.output_dir}/runtime_bundle",
        "--calibration",
        f"{args.output_dir}/calibration.json",
        "--metrics",
        f"{args.output_dir}/eval_metrics.json",
        "--training-config",
        f"{args.output_dir}/training_config.json",
        "--export-summary",
        f"{args.output_dir}/export_summary.json",
    ]
    if args.quantize:
        export_cmd.append("--quantize")
    run_step(export_cmd, root)
    run_step([sys.executable, "-m", "src.model_card", "--artifact-dir", args.output_dir], root)


if __name__ == "__main__":
    main()
