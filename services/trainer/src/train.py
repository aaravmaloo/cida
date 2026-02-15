from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.constants import ARTIFACT_ROOT
from src.data_pipeline import load_and_split, normalize_text


class WeightedBCETrainer(Trainer):
    def __init__(self, pos_weight: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.view(-1)
        labels = labels.float()

        weight = self.pos_weight.to(logits.device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def head_tail_tokenize(tokenizer, text: str, max_len: int = 512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_len - 2:
        chunk = tokens
    else:
        head = (max_len - 2) // 2
        tail = max_len - 2 - head
        chunk = tokens[:head] + tokens[-tail:]
    input_ids = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)

    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits.reshape(-1)
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
    }


def run(args: argparse.Namespace) -> None:
    artifact_dir = Path(args.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    split = load_and_split(args.csv, text_col=args.text_col, label_col=args.label_col, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def to_dataset(frame):
        text = frame[args.text_col].apply(normalize_text).tolist()
        labels = frame[args.label_col].tolist()
        encoded = [head_tail_tokenize(tokenizer, t, max_len=args.max_len) for t in text]
        return Dataset.from_dict(
            {
                "input_ids": [x["input_ids"] for x in encoded],
                "attention_mask": [x["attention_mask"] for x in encoded],
                "labels": labels,
            }
        )

    train_ds = to_dataset(split.train)
    val_ds = to_dataset(split.val)
    test_ds = to_dataset(split.test)

    pos = int(split.train[args.label_col].sum())
    neg = int(len(split.train) - pos)
    pos_weight = neg / max(1, pos)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)

    train_args = TrainingArguments(
        output_dir=str(artifact_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        load_best_model_at_end=True,
        logging_steps=20,
        report_to=[],
        fp16=torch.cuda.is_available(),
        seed=args.seed,
    )

    trainer = WeightedBCETrainer(
        pos_weight=pos_weight,
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    val_pred = trainer.predict(val_ds)
    test_pred = trainer.predict(test_ds)

    final_metrics = {
        "val": compute_metrics((val_pred.predictions, val_pred.label_ids)),
        "test": compute_metrics((test_pred.predictions, test_pred.label_ids)),
        "model_name": args.model_name,
        "max_len": args.max_len,
    }

    model_path = artifact_dir / "model"
    trainer.model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    np.savez(
        artifact_dir / "val_logits.npz",
        logits=val_pred.predictions.reshape(-1),
        labels=val_pred.label_ids,
    )
    np.savez(
        artifact_dir / "test_logits.npz",
        logits=test_pred.predictions.reshape(-1),
        labels=test_pred.label_ids,
    )

    (artifact_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../../train_data/balanced_ai_human_prompts.csv")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="generated")
    parser.add_argument("--output-dir", default=str(ARTIFACT_ROOT))
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())

