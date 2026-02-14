from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cida.data import prepare_dataset


def _add_prepare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-csv", required=True, help="Path to CSV with text + label columns.")
    parser.add_argument("--output-dir", default="artifacts", help="Where tokenizer + encoded splits are saved.")
    parser.add_argument("--text-col", default="text", help="Text column name.")
    parser.add_argument(
        "--label-col",
        default="generated",
        help="Binary label column (1=AI, 0=Human).",
    )
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--min-pair-freq", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lowercase", action="store_true", help="Lowercase text before tokenization.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="CPU workers for encoding. 0 = all cores.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Work chunk size per worker process (auto if omitted).",
    )
    parser.add_argument(
        "--compression",
        choices=["none", "compressed"],
        default="none",
        help="Artifact compression mode. 'none' is faster; 'compressed' is smaller.",
    )


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["torch", "tensorflow"], required=True)
    parser.add_argument("--artifact-dir", default="artifacts", help="Path created by `prepare`.")
    parser.add_argument("--output-dir", default="runs/latest", help="Training output directory.")

    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--cls-hidden-dim", type=int, default=256)

    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch only: auto/cpu/cuda/cuda:0. Ignored for tensorflow.",
    )


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["torch", "tensorflow"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Torch: best.pt | TensorFlow: checkpoint.json")
    parser.add_argument("--split-path", required=True, help="Encoded split .npz (val/test).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="auto", help="Torch only.")
    parser.add_argument("--output-path", default=None, help="Optional JSON metrics output path.")


def _add_predict_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["torch", "tensorflow"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Torch: best.pt | TensorFlow: checkpoint.json")
    parser.add_argument("--text", default=None, help="Single text input.")
    parser.add_argument("--input-file", default=None, help="Text file (1 sample per line).")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64, help="TensorFlow predict batch size.")
    parser.add_argument("--device", default="auto", help="Torch only.")
    parser.add_argument("--output-file", default=None, help="Optional JSONL output path.")


def _load_texts(text: str | None, input_file: str | None) -> list[str]:
    if text and input_file:
        raise ValueError("Use either --text or --input-file, not both.")
    if not text and not input_file:
        raise ValueError("Provide --text or --input-file.")
    if text:
        return [text]
    lines = Path(input_file).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _train_from_args(args: argparse.Namespace) -> dict:
    metadata_path = Path(args.artifact_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Run `py cida_dev.py prepare --input-csv train_data/balanced_ai_human_prompts.csv` "
            "or use `py cida_dev.py fit ...`."
        )

    common = dict(
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        pooling=args.pooling,
        cls_hidden_dim=args.cls_hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        threshold=args.threshold,
        seed=args.seed,
    )
    if args.backend == "torch":
        from cida.torch_backend import train_torch

        return train_torch(
            **common,
            grad_clip=args.grad_clip,
            device=args.device,
        )
    from cida.tf_backend import train_tensorflow

    return train_tensorflow(**common)


def _evaluate_from_args(args: argparse.Namespace) -> dict:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    split_path = Path(args.split_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Encoded split not found: {split_path}")

    if args.backend == "torch":
        from cida.torch_backend import evaluate_torch

        return evaluate_torch(
            checkpoint_path=args.checkpoint,
            split_path=args.split_path,
            batch_size=args.batch_size,
            threshold=args.threshold,
            device=args.device,
            output_path=args.output_path,
        )
    from cida.tf_backend import evaluate_tensorflow

    return evaluate_tensorflow(
        checkpoint_json=args.checkpoint,
        split_path=args.split_path,
        batch_size=args.batch_size,
        threshold=args.threshold,
        output_path=args.output_path,
    )


def _predict_from_args(args: argparse.Namespace) -> list[dict]:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    texts = _load_texts(args.text, args.input_file)
    if args.backend == "torch":
        from cida.torch_backend import predict_torch

        return predict_torch(
            checkpoint_path=args.checkpoint,
            texts=texts,
            threshold=args.threshold,
            device=args.device,
        )
    from cida.tf_backend import predict_tensorflow

    return predict_tensorflow(
        checkpoint_json=args.checkpoint,
        texts=texts,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )


def _save_jsonl(rows: list[dict], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cida_dev",
        description="CLI for AI text detection using scratch-trained Transformer encoders.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Train tokenizer + create train/val/test encoded artifacts.")
    _add_prepare_args(p_prepare)

    p_train = sub.add_parser("train", help="Train the model with torch or tensorflow backend.")
    _add_train_args(p_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate a trained checkpoint on encoded data.")
    _add_eval_args(p_eval)

    p_predict = sub.add_parser("predict", help="Predict AI probability for text inputs.")
    _add_predict_args(p_predict)

    p_fit = sub.add_parser("fit", help="Run prepare -> train -> evaluate in one command.")
    p_fit.add_argument("--input-csv", required=True, help="Path to CSV with text + label columns.")
    p_fit.add_argument("--artifact-dir", default="artifacts", help="Where tokenizer + encoded splits are saved.")
    p_fit.add_argument("--run-dir", default="runs/latest", help="Where trained model artifacts are saved.")
    p_fit.add_argument("--text-col", default="text", help="Text column name.")
    p_fit.add_argument("--label-col", default="generated", help="Binary label column (1=AI, 0=Human).")
    p_fit.add_argument("--vocab-size", type=int, default=8000)
    p_fit.add_argument("--min-pair-freq", type=int, default=2)
    p_fit.add_argument("--max-len", type=int, default=512)
    p_fit.add_argument("--val-ratio", type=float, default=0.1)
    p_fit.add_argument("--test-ratio", type=float, default=0.1)
    p_fit.add_argument("--lowercase", action="store_true", help="Lowercase text before tokenization.")
    p_fit.add_argument("--workers", type=int, default=0, help="CPU workers for encoding. 0 = all cores.")
    p_fit.add_argument("--chunk-size", type=int, default=None, help="Work chunk size per worker process.")
    p_fit.add_argument(
        "--compression",
        choices=["none", "compressed"],
        default="none",
        help="Artifact compression mode. 'none' is faster; 'compressed' is smaller.",
    )

    p_fit.add_argument("--backend", choices=["torch", "tensorflow"], required=True)
    p_fit.add_argument("--d-model", type=int, default=384)
    p_fit.add_argument("--n-heads", type=int, default=8)
    p_fit.add_argument("--n-layers", type=int, default=6)
    p_fit.add_argument("--ffn-dim", type=int, default=1024)
    p_fit.add_argument("--dropout", type=float, default=0.1)
    p_fit.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    p_fit.add_argument("--cls-hidden-dim", type=int, default=256)
    p_fit.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=20)
    p_fit.add_argument("--batch-size", type=int, default=24)
    p_fit.add_argument("--learning-rate", type=float, default=3e-4)
    p_fit.add_argument("--weight-decay", type=float, default=0.01)
    p_fit.add_argument("--warmup-steps", type=int, default=200)
    p_fit.add_argument("--grad-clip", type=float, default=1.0)
    p_fit.add_argument("--label-smoothing", type=float, default=0.0)
    p_fit.add_argument("--patience", type=int, default=5)
    p_fit.add_argument("--threshold", type=float, default=0.5)
    p_fit.add_argument("--seed", type=int, default=42)
    p_fit.add_argument("--device", default="auto", help="Torch only: auto/cpu/cuda/cuda:0.")
    p_fit.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default="test",
        help="Split to evaluate after training.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        metadata = prepare_dataset(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            text_col=args.text_col,
            label_col=args.label_col,
            vocab_size=args.vocab_size,
            min_pair_freq=args.min_pair_freq,
            max_len=args.max_len,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            lowercase=args.lowercase,
            workers=args.workers,
            chunk_size=args.chunk_size,
            compression=args.compression,
        )
        print(json.dumps(metadata, ensure_ascii=True, indent=2))
        return

    if args.command == "train":
        summary = _train_from_args(args)
        print(json.dumps(summary, ensure_ascii=True, indent=2))
        return

    if args.command == "evaluate":
        metrics = _evaluate_from_args(args)
        print(json.dumps(metrics, ensure_ascii=True, indent=2))
        return

    if args.command == "predict":
        preds = _predict_from_args(args)
        if args.output_file:
            _save_jsonl(preds, args.output_file)
        print(json.dumps(preds, ensure_ascii=True, indent=2))
        return

    if args.command == "fit":
        metadata = prepare_dataset(
            input_csv=args.input_csv,
            output_dir=args.artifact_dir,
            text_col=args.text_col,
            label_col=args.label_col,
            vocab_size=args.vocab_size,
            min_pair_freq=args.min_pair_freq,
            max_len=args.max_len,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            lowercase=args.lowercase,
            workers=args.workers,
            chunk_size=args.chunk_size,
            compression=args.compression,
        )
        fit_train_args = argparse.Namespace(
            backend=args.backend,
            artifact_dir=args.artifact_dir,
            output_dir=args.run_dir,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
            pooling=args.pooling,
            cls_hidden_dim=args.cls_hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing,
            patience=args.patience,
            threshold=args.threshold,
            seed=args.seed,
            device=args.device,
        )
        summary = _train_from_args(fit_train_args)

        eval_split = args.eval_split
        split_path = Path(args.artifact_dir) / "data" / f"{eval_split}.npz"
        if args.backend == "torch":
            from cida.torch_backend import evaluate_torch

            checkpoint = Path(args.run_dir) / "best.pt"
            metrics = evaluate_torch(
                checkpoint_path=checkpoint,
                split_path=split_path,
                batch_size=max(32, args.batch_size),
                threshold=None,
                device=args.device,
                output_path=Path(args.run_dir) / f"{eval_split}_metrics.json",
            )
        else:
            from cida.tf_backend import evaluate_tensorflow

            checkpoint_json = Path(args.run_dir) / "checkpoint.json"
            metrics = evaluate_tensorflow(
                checkpoint_json=checkpoint_json,
                split_path=split_path,
                batch_size=max(32, args.batch_size),
                threshold=None,
                output_path=Path(args.run_dir) / f"{eval_split}_metrics.json",
            )

        result = {"prepare": metadata, "train": summary, "evaluate": metrics}
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
