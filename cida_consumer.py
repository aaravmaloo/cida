from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_texts(text: str | None, input_file: str | None) -> list[str]:
    if text and input_file:
        raise ValueError("Use either --text or --input-file, not both.")
    if not text and not input_file:
        raise ValueError("Provide --text or --input-file.")
    if text:
        return [text]
    lines = Path(input_file).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _save_jsonl(rows: list[dict], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cida",
        description="Consumer CLI for AI-text detection using TensorFlow .h5 model artifacts.",
    )
    parser.add_argument("--bundle", default=None, help="Path to consumer_config.json from training output.")
    parser.add_argument("--model-h5", default=None, help="Path to TensorFlow .h5 model.")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer.json (required without --bundle).")
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Max token length used during training (required without --bundle).",
    )
    parser.add_argument("--text", default=None, help="Single text input.")
    parser.add_argument("--input-file", default=None, help="Text file with one sample per line.")
    parser.add_argument("--threshold", type=float, default=None, help="Override classification threshold.")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--output-file", default=None, help="Optional JSONL output path.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    texts = _load_texts(args.text, args.input_file)
    if args.bundle:
        from cida.consumer import predict_from_bundle

        preds = predict_from_bundle(
            bundle_config_path=args.bundle,
            texts=texts,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )
    else:
        if not args.model_h5 or not args.tokenizer or args.max_len is None:
            raise ValueError(
                "Without --bundle, you must provide --model-h5, --tokenizer, and --max-len."
            )
        from cida.consumer import predict_h5

        thr = 0.5 if args.threshold is None else args.threshold
        preds = predict_h5(
            model_h5_path=args.model_h5,
            tokenizer_path=args.tokenizer,
            texts=texts,
            max_len=args.max_len,
            threshold=thr,
            batch_size=args.batch_size,
        )

    if args.output_file:
        _save_jsonl(preds, args.output_file)
    print(json.dumps(preds, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
