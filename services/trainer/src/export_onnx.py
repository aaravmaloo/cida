from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def export_onnx(model_dir: Path, output: Path, max_len: int) -> None:
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.eval()
    dummy = tokenizer(
        "This is a sample input for ONNX export.",
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        output,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )


def quantize_onnx(input_path: Path, output_path: Path) -> bool:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
        )
        return True
    except Exception:
        return False


def build_runtime_bundle(
    *,
    bundle_dir: Path,
    model_dir: Path,
    onnx_path: Path,
    calibration_path: Path | None,
    metrics_path: Path | None,
    training_config_path: Path | None,
) -> None:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(onnx_path, bundle_dir / "model.onnx")

    tokenizer_dir = bundle_dir / "model"
    shutil.copytree(model_dir, tokenizer_dir)

    if calibration_path and calibration_path.exists():
        shutil.copy2(calibration_path, bundle_dir / "calibration.json")
    if metrics_path and metrics_path.exists():
        shutil.copy2(metrics_path, bundle_dir / "metrics.json")
    if training_config_path and training_config_path.exists():
        shutil.copy2(training_config_path, bundle_dir / "training_config.json")

    manifest = {
        "onnx": "model.onnx",
        "tokenizer_dir": "model",
        "calibration": "calibration.json" if (bundle_dir / "calibration.json").exists() else None,
        "metrics": "metrics.json" if (bundle_dir / "metrics.json").exists() else None,
        "training_config": "training_config.json" if (bundle_dir / "training_config.json").exists() else None,
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    output = Path(args.output)

    export_onnx(model_dir=model_dir, output=output, max_len=args.max_len)

    quantized = False
    if args.quantize:
        quantized_path = output.with_name(f"{output.stem}.int8{output.suffix}")
        quantized = quantize_onnx(output, quantized_path)
        if quantized:
            shutil.move(str(quantized_path), str(output))

    if args.bundle_dir:
        build_runtime_bundle(
            bundle_dir=Path(args.bundle_dir),
            model_dir=model_dir,
            onnx_path=output,
            calibration_path=Path(args.calibration) if args.calibration else None,
            metrics_path=Path(args.metrics) if args.metrics else None,
            training_config_path=Path(args.training_config) if args.training_config else None,
        )

    if args.export_summary:
        summary = {
            "onnx_path": str(output),
            "quantized": bool(quantized),
            "bundle_dir": args.bundle_dir or "",
        }
        Path(args.export_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="../artifacts/latest/model")
    parser.add_argument("--output", default="../artifacts/latest/model.onnx")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--bundle-dir", default="../artifacts/latest/runtime_bundle")
    parser.add_argument("--calibration", default="../artifacts/latest/calibration.json")
    parser.add_argument("--metrics", default="../artifacts/latest/eval_metrics.json")
    parser.add_argument("--training-config", default="../artifacts/latest/training_config.json")
    parser.add_argument("--export-summary", default="../artifacts/latest/export_summary.json")
    run(parser.parse_args())
