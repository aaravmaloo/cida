from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.constants import ARTIFACT_ROOT
from src.data_pipeline import load_and_split, normalize_text


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def safe_roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    # roc_auc_score fails when a split has one class.
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, probs))


def compute_metrics_from_logits(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    logits = logits.reshape(-1).astype(np.float64)
    labels = labels.reshape(-1).astype(np.int64)
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(np.int64)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "roc_auc": safe_roc_auc(labels, probs),
        "brier": float(brier_score_loss(labels, probs)),
    }


def find_optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> dict:
    labels = labels.reshape(-1).astype(np.int64)
    probs = probs.reshape(-1).astype(np.float64)

    best = {"threshold": 0.5, "f1": -1.0, "accuracy": -1.0, "precision": 0.0, "recall": 0.0}
    for t in np.linspace(0.05, 0.95, 181):
        preds = (probs >= t).astype(np.int64)
        f1 = float(f1_score(labels, preds, zero_division=0))
        acc = float(accuracy_score(labels, preds))
        if f1 > best["f1"] or (abs(f1 - best["f1"]) <= 1e-9 and acc > best["accuracy"]):
            best = {
                "threshold": float(t),
                "f1": f1,
                "accuracy": acc,
                "precision": float(precision_score(labels, preds, zero_division=0)),
                "recall": float(recall_score(labels, preds, zero_division=0)),
            }
    return best


class WeightedFocalBCETrainer(Trainer):
    def __init__(self, pos_weight: float, focal_gamma: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
        self.focal_gamma = float(max(0.0, focal_gamma))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.view(-1)
        labels = labels.float()

        pos_weight = self.pos_weight.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=pos_weight)
        if self.focal_gamma > 0:
            probs = torch.sigmoid(logits)
            pt = torch.where(labels > 0.5, probs, 1.0 - probs)
            focal = (1.0 - pt).pow(self.focal_gamma)
            loss = (focal * bce).mean()
        else:
            loss = bce.mean()

        return (loss, outputs) if return_outputs else loss


def head_tail_tokenize(tokenizer, text: str, max_len: int = 512) -> dict:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_len - 2:
        chunk = token_ids
    else:
        head = (max_len - 2) // 2
        tail = max_len - 2 - head
        chunk = token_ids[:head] + token_ids[-tail:]

    input_ids = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)

    pad = max_len - len(input_ids)
    if pad > 0:
        input_ids += [tokenizer.pad_token_id] * pad
        attention_mask += [0] * pad

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def to_dataset(frame, tokenizer, text_col: str, label_col: str, max_len: int) -> Dataset:
    texts = frame[text_col].apply(normalize_text).tolist()
    labels = frame[label_col].astype(int).tolist()
    encoded = [head_tail_tokenize(tokenizer, text, max_len=max_len) for text in texts]
    return Dataset.from_dict(
        {
            "input_ids": [item["input_ids"] for item in encoded],
            "attention_mask": [item["attention_mask"] for item in encoded],
            "labels": labels,
        }
    )


def build_model(args: argparse.Namespace, tokenizer) -> tuple[torch.nn.Module, int, dict]:
    if args.from_scratch:
        if args.hidden_size % args.attention_heads != 0:
            raise ValueError("hidden-size must be divisible by attention-heads")

        config = BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.layers,
            num_attention_heads=args.attention_heads,
            intermediate_size=args.ffn_size,
            max_position_embeddings=max(args.max_position_embeddings, args.max_len + 2),
            type_vocab_size=2,
            num_labels=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = BertForSequenceClassification(config)
        model_spec = {
            "mode": "from_scratch_bert",
            "num_hidden_layers": int(args.layers),
            "hidden_size": int(args.hidden_size),
            "num_attention_heads": int(args.attention_heads),
            "intermediate_size": int(args.ffn_size),
            "max_position_embeddings": int(config.max_position_embeddings),
        }
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
        model_spec = {"mode": "pretrained", "model_name": args.model_name}

    param_count = int(sum(p.numel() for p in model.parameters()))
    if args.min_params > 0 and param_count < args.min_params:
        raise ValueError(
            f"Model has {param_count:,} parameters, below required minimum {args.min_params:,}. "
            "Increase hidden-size/ffn-size/layers or lower --min-params."
        )
    return model, param_count, model_spec


def detect_accelerator() -> dict:
    info: dict[str, object] = {
        "accelerator": "cpu",
        "xla_hardware": None,
        "xla_world_size": 0,
        "cuda_devices": 0,
        "cuda_name": None,
        "cuda_bf16_supported": False,
    }

    if torch.cuda.is_available():
        info["accelerator"] = "cuda"
        info["cuda_devices"] = int(torch.cuda.device_count())
        props = torch.cuda.get_device_properties(0)
        info["cuda_name"] = props.name
        bf16_supported = False
        if hasattr(torch.cuda, "is_bf16_supported"):
            bf16_supported = bool(torch.cuda.is_bf16_supported())
        info["cuda_bf16_supported"] = bf16_supported
        return info

    if importlib.util.find_spec("torch_xla") is None:
        return info

    try:
        import torch_xla.core.xla_model as xm

        xla_device = xm.xla_device()
        hardware = str(xm.xla_device_hw(xla_device)).upper()
        info["xla_hardware"] = hardware
        info["xla_world_size"] = int(xm.xrt_world_size())
        if hardware == "TPU":
            info["accelerator"] = "tpu"
        else:
            info["accelerator"] = f"xla_{hardware.lower()}" if hardware else "xla"
    except Exception:
        return info

    return info


def resolve_precision(args: argparse.Namespace, accelerator_info: dict) -> dict:
    accelerator = str(accelerator_info.get("accelerator", "cpu"))
    use_cuda = accelerator == "cuda"
    use_tpu = accelerator == "tpu"
    cuda_bf16_supported = bool(accelerator_info.get("cuda_bf16_supported", False))

    use_tpu_bf16 = bool(use_tpu and args.tpu_bf16)
    use_gpu_bf16 = bool(use_cuda and args.gpu_bf16 and cuda_bf16_supported)
    use_gpu_fp16 = bool(use_cuda and args.gpu_fp16 and not use_gpu_bf16)
    use_gpu_tf32 = bool(use_cuda and args.gpu_tf32)

    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = use_gpu_tf32
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = use_gpu_tf32

    return {
        "fp16": use_gpu_fp16,
        "bf16": use_tpu_bf16 or use_gpu_bf16,
        "tf32": use_gpu_tf32,
        "tpu_bf16": use_tpu_bf16,
        "gpu_bf16": use_gpu_bf16,
        "gpu_fp16": use_gpu_fp16,
        "cuda_bf16_supported": cuda_bf16_supported,
    }


def run(args: argparse.Namespace) -> None:
    artifact_dir = Path(args.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if args.data_paths:
        data_files = args.data_paths
    else:
        data_files = [args.csv]

    split = load_and_split(
        data_files,
        text_col=args.text_col,
        label_col=args.label_col,
        seed=args.seed,
        unlabeled_default_label=args.unlabeled_default_label,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = to_dataset(split.train, tokenizer, split.text_col, split.label_col, args.max_len)
    val_ds = to_dataset(split.val, tokenizer, split.text_col, split.label_col, args.max_len)
    test_ds = to_dataset(split.test, tokenizer, split.text_col, split.label_col, args.max_len)

    pos = int(split.train[split.label_col].sum())
    neg = int(len(split.train) - pos)
    pos_weight = float(neg / max(1, pos))

    model, param_count, model_spec = build_model(args, tokenizer)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    accelerator_info = detect_accelerator()
    accelerator = str(accelerator_info["accelerator"])
    use_cuda = accelerator == "cuda"
    use_tpu = accelerator == "tpu"
    tpu_cores = int(accelerator_info.get("xla_world_size", 0)) if use_tpu else None
    precision = resolve_precision(args, accelerator_info)

    train_args = TrainingArguments(
        output_dir=str(artifact_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        metric_for_best_model="eval_roc_auc",
        greater_is_better=True,
        load_best_model_at_end=True,
        logging_steps=20,
        report_to=[],
        fp16=bool(precision["fp16"]),
        bf16=bool(precision["bf16"]),
        dataloader_pin_memory=use_cuda,
        tpu_num_cores=tpu_cores,
        seed=args.seed,
    )

    def trainer_metrics(eval_pred):
        logits, labels = eval_pred
        return compute_metrics_from_logits(logits, labels, threshold=0.5)

    trainer = WeightedFocalBCETrainer(
        pos_weight=pos_weight,
        focal_gamma=args.focal_gamma,
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    val_pred = trainer.predict(val_ds)
    test_pred = trainer.predict(test_ds)

    val_logits = val_pred.predictions.reshape(-1).astype(np.float64)
    val_labels = val_pred.label_ids.reshape(-1).astype(np.int64)
    test_logits = test_pred.predictions.reshape(-1).astype(np.float64)
    test_labels = test_pred.label_ids.reshape(-1).astype(np.int64)

    threshold_info = find_optimal_threshold(sigmoid(val_logits), val_labels)
    threshold = float(threshold_info["threshold"])

    final_metrics = {
        "model_name": args.model_name,
        "model_spec": model_spec,
        "parameter_count": param_count,
        "accelerator": accelerator_info,
        "precision": precision,
        "data_files": data_files,
        "text_col": split.text_col,
        "label_col": split.label_col,
        "max_len": args.max_len,
        "split_sizes": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "train_class_balance": {"human_0": int(neg), "ai_1": int(pos)},
        "loss": {"pos_weight": pos_weight, "focal_gamma": float(args.focal_gamma)},
        "val@0.5": compute_metrics_from_logits(val_logits, val_labels, threshold=0.5),
        "val@best": compute_metrics_from_logits(val_logits, val_labels, threshold=threshold),
        "test@0.5": compute_metrics_from_logits(test_logits, test_labels, threshold=0.5),
        "test@best": compute_metrics_from_logits(test_logits, test_labels, threshold=threshold),
        "best_threshold_from_val": threshold_info,
    }

    model_path = artifact_dir / "model"
    trainer.model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    np.savez(artifact_dir / "val_logits.npz", logits=val_logits, labels=val_labels)
    np.savez(artifact_dir / "test_logits.npz", logits=test_logits, labels=test_labels)

    training_config = {
        "data_files": data_files,
        "csv": args.csv if not args.data_paths else None,
        "text_col": split.text_col,
        "label_col": split.label_col,
        "requested_label_col": args.label_col,
        "unlabeled_default_label": args.unlabeled_default_label,
        "model_name": args.model_name,
        "model_spec": model_spec,
        "parameter_count": param_count,
        "min_params": args.min_params,
        "from_scratch": bool(args.from_scratch),
        "accelerator": accelerator_info,
        "precision": precision,
        "tpu_bf16": bool(args.tpu_bf16),
        "gpu_bf16": bool(args.gpu_bf16),
        "gpu_fp16": bool(args.gpu_fp16),
        "gpu_tf32": bool(args.gpu_tf32),
        "layers": args.layers,
        "hidden_size": args.hidden_size,
        "attention_heads": args.attention_heads,
        "ffn_size": args.ffn_size,
        "max_position_embeddings": args.max_position_embeddings,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_accum": args.grad_accum,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler": args.lr_scheduler,
        "focal_gamma": args.focal_gamma,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "seed": args.seed,
    }

    (artifact_dir / "thresholds.json").write_text(json.dumps({"optimal_threshold": threshold, **threshold_info}, indent=2), encoding="utf-8")
    (artifact_dir / "training_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")
    (artifact_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", dest="data_paths", action="append", help="Training file path (.csv/.parquet). Repeat to include multiple files.")
    parser.add_argument("--csv", default="../../train_data/balanced_ai_human_prompts.csv")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="auto")
    parser.add_argument("--unlabeled-default-label", type=int, choices=[0, 1], default=None)
    parser.add_argument("--output-dir", default=str(ARTIFACT_ROOT))
    parser.add_argument("--model-name", default="microsoft/deberta-v3-large")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--attention-heads", type=int, default=16)
    parser.add_argument("--ffn-size", type=int, default=4096)
    parser.add_argument("--max-position-embeddings", type=int, default=514)
    parser.add_argument("--min-params", type=int, default=100_000_000)
    parser.add_argument("--tpu-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
