from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def run(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.eval()
    dummy = tokenizer(
        "This is a sample input for ONNX export.",
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="../artifacts/latest/model")
    parser.add_argument("--output", default="../artifacts/latest/model.onnx")
    run(parser.parse_args())

