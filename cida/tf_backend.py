from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from cida.data import load_npz_split
from cida.metrics import best_f1_threshold, binary_metrics, save_metrics
from cida.tokenizer import BPETokenizer

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - runtime dependency gate
    tf = None


def _require_tf() -> None:
    if tf is None:
        raise ImportError("TensorFlow is not installed. Install tensorflow to use --backend tensorflow.")


@dataclass
class TFModelConfig:
    vocab_size: int
    max_len: int
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    ffn_dim: int = 1024
    dropout: float = 0.1
    pooling: str = "cls"
    cls_hidden_dim: int = 256


if tf is not None:

    class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):  # type: ignore[misc]
        def __init__(self, base_lr: float, warmup_steps: int, total_steps: int):
            super().__init__()
            self.base_lr = float(base_lr)
            self.warmup_steps = int(max(0, warmup_steps))
            self.total_steps = int(max(1, total_steps))

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            warmup_steps = tf.cast(self.warmup_steps, tf.float32)
            total_steps = tf.cast(self.total_steps, tf.float32)

            warmup_lr = self.base_lr * step / tf.maximum(1.0, warmup_steps)
            progress = (step - warmup_steps) / tf.maximum(1.0, total_steps - warmup_steps)
            progress = tf.clip_by_value(progress, 0.0, 1.0)
            cosine_lr = self.base_lr * 0.5 * (1.0 + tf.cos(math.pi * progress))
            return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

        def get_config(self):
            return {
                "base_lr": self.base_lr,
                "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps,
            }
else:

    class WarmupCosineSchedule:  # pragma: no cover - runtime dependency gate
        def __init__(self, *args, **kwargs):
            _require_tf()


def _build_tf_model(cfg: TFModelConfig) -> tf.keras.Model:
    max_len = int(cfg.max_len)
    input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    tok_emb = tf.keras.layers.Embedding(cfg.vocab_size, cfg.d_model, name="token_embedding")(input_ids)

    pos_ids = tf.keras.layers.Lambda(
        lambda t: tf.tile(tf.range(max_len, dtype=tf.int32)[tf.newaxis, :], [tf.shape(t)[0], 1]),
        name="position_ids",
    )(input_ids)
    pos_emb = tf.keras.layers.Embedding(max_len, cfg.d_model, name="position_embedding")(pos_ids)

    x = tf.keras.layers.Add(name="input_add")([tok_emb, pos_emb])
    x = tf.keras.layers.Dropout(cfg.dropout, name="input_dropout")(x)

    attn_mask = tf.keras.layers.Lambda(
        lambda m: tf.cast(m[:, tf.newaxis, :], tf.bool),
        name="attention_mask_expanded",
    )(attention_mask)

    for i in range(cfg.n_layers):
        ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"ln1_{i}")(x)
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=cfg.n_heads,
            key_dim=cfg.d_model // cfg.n_heads,
            dropout=cfg.dropout,
            name=f"mha_{i}",
        )(ln1, ln1, attention_mask=attn_mask)
        attn = tf.keras.layers.Dropout(cfg.dropout, name=f"attn_drop_{i}")(attn)
        x = tf.keras.layers.Add(name=f"attn_residual_{i}")([x, attn])

        ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"ln2_{i}")(x)
        ffn = tf.keras.layers.Dense(cfg.ffn_dim, activation="gelu", name=f"ffn_up_{i}")(ln2)
        ffn = tf.keras.layers.Dropout(cfg.dropout, name=f"ffn_drop1_{i}")(ffn)
        ffn = tf.keras.layers.Dense(cfg.d_model, name=f"ffn_down_{i}")(ffn)
        ffn = tf.keras.layers.Dropout(cfg.dropout, name=f"ffn_drop2_{i}")(ffn)
        x = tf.keras.layers.Add(name=f"ffn_residual_{i}")([x, ffn])

    x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_ln")(x)

    if cfg.pooling == "mean":
        mask_f = tf.keras.layers.Lambda(
            lambda m: tf.cast(tf.expand_dims(m, axis=-1), tf.float32),
            name="mask_float",
        )(attention_mask)
        masked_sum = tf.keras.layers.Lambda(
            lambda t: tf.reduce_sum(t[0] * t[1], axis=1),
            name="masked_sum",
        )([x, mask_f])
        denom = tf.keras.layers.Lambda(
            lambda m: tf.maximum(1.0, tf.reduce_sum(m, axis=1)),
            name="mask_denom",
        )(mask_f)
        pooled = tf.keras.layers.Lambda(lambda t: t[0] / t[1], name="mean_pool")([masked_sum, denom])
    else:
        pooled = tf.keras.layers.Lambda(lambda t: t[:, 0, :], name="cls_pool")(x)

    hidden = tf.keras.layers.Dense(cfg.cls_hidden_dim, activation="relu", name="cls_hidden")(pooled)
    hidden = tf.keras.layers.Dropout(cfg.dropout, name="cls_drop")(hidden)
    logits = tf.keras.layers.Dense(1, name="logits")(hidden)
    logits = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="squeeze")(logits)

    return tf.keras.Model(inputs=[input_ids, attention_mask], outputs=logits, name="cida_tf_transformer")


def _build_dataset(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (
            {"input_ids": input_ids.astype(np.int32), "attention_mask": attention_mask.astype(np.int32)},
            labels.astype(np.float32),
        )
    )
    if training:
        ds = ds.shuffle(buffer_size=len(labels), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def train_tensorflow(
    artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    d_model: int = 384,
    n_heads: int = 8,
    n_layers: int = 6,
    ffn_dim: int = 1024,
    dropout: float = 0.1,
    pooling: str = "cls",
    cls_hidden_dim: int = 256,
    epochs: int = 20,
    batch_size: int = 24,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 200,
    label_smoothing: float = 0.0,
    patience: int = 5,
    threshold: float = 0.5,
    seed: int = 42,
) -> dict:
    _require_tf()
    tf.keras.utils.set_random_seed(seed)

    artifact_dir = Path(artifact_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    tokenizer = BPETokenizer.load(artifact_dir / "tokenizer" / "tokenizer.json")

    train_ids, train_mask, train_y = load_npz_split(artifact_dir / "data" / "train.npz")
    val_ids, val_mask, val_y = load_npz_split(artifact_dir / "data" / "val.npz")

    cfg = TFModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_len=int(metadata["max_len"]),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
        pooling=pooling,
        cls_hidden_dim=cls_hidden_dim,
    )
    model = _build_tf_model(cfg)

    train_ds = _build_dataset(train_ids, train_mask, train_y, batch_size=batch_size, training=True)
    val_ds = _build_dataset(val_ids, val_mask, val_y, batch_size=batch_size * 2, training=False)

    steps_per_epoch = max(1, int(math.ceil(len(train_y) / batch_size)))
    total_steps = max(1, steps_per_epoch * epochs)
    lr_schedule = WarmupCosineSchedule(learning_rate, warmup_steps=warmup_steps, total_steps=total_steps)

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
    )
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=label_smoothing)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks,
    )

    # Save history in CSV.
    history_rows = []
    epochs_ran = len(history.history.get("loss", []))
    for i in range(epochs_ran):
        history_rows.append(
            {
                "epoch": i + 1,
                "train_loss": float(history.history.get("loss", [])[i]),
                "val_loss": float(history.history.get("val_loss", [])[i]) if "val_loss" in history.history else float("nan"),
                "train_auc": float(history.history.get("auc", [])[i]) if "auc" in history.history else float("nan"),
                "val_auc": float(history.history.get("val_auc", [])[i]) if "val_auc" in history.history else float("nan"),
            }
        )

    with (out_dir / "train_history.csv").open("w", encoding="utf-8", newline="") as handle:
        if history_rows:
            writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)


    val_logits = model.predict(
        {"input_ids": val_ids.astype(np.int32), "attention_mask": val_mask.astype(np.int32)},
        batch_size=batch_size * 2,
        verbose=0,
    )
    val_prob = 1.0 / (1.0 + np.exp(-val_logits.reshape(-1)))
    tuned_threshold, tuned_metrics = best_f1_threshold(val_y, val_prob)
    default_metrics = binary_metrics(val_y, val_prob, threshold=threshold)

    best_model_path = out_dir / "best.h5"
    model.save(best_model_path, include_optimizer=False)
    tokenizer_export_path = out_dir / "tokenizer.json"
    tokenizer_export_path.write_text(
        json.dumps(tokenizer.to_dict(), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    consumer_config = {
        "model_path": "best.h5",
        "tokenizer_path": "tokenizer.json",
        "max_len": int(cfg.max_len),
        "threshold": float(tuned_threshold),
        "label_mapping": {"0": "human", "1": "ai"},
    }
    save_metrics(consumer_config, out_dir / "consumer_config.json")

    checkpoint = {
        "backend": "tensorflow",
        "model_path": "best.h5",
        "model_config": asdict(cfg),
        "train_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "label_smoothing": label_smoothing,
            "patience": patience,
            "seed": seed,
        },
        "metadata": metadata,
        "tokenizer": tokenizer.to_dict(),
        "best_threshold": float(tuned_threshold),
        "best_val_metrics": tuned_metrics,
        "default_val_metrics": default_metrics,
    }
    save_metrics(checkpoint, out_dir / "checkpoint.json")

    summary = {
        "backend": "tensorflow",
        "best_threshold": float(tuned_threshold),
        "best_val_metrics": tuned_metrics,
        "default_val_metrics": default_metrics,
        "output_dir": str(out_dir),
        "model_path": str(best_model_path),
        "consumer_config": str(out_dir / "consumer_config.json"),
        "consumer_tokenizer": str(tokenizer_export_path),
    }
    save_metrics(summary, out_dir / "train_summary.json")
    return summary


def _load_tf_model_file(model_path: str | Path):
    _require_tf()
    path = Path(model_path)
    try:
        return tf.keras.models.load_model(path, compile=False, safe_mode=False)
    except TypeError:
        # Older TF/Keras versions without `safe_mode`.
        return tf.keras.models.load_model(path, compile=False)


def _load_tf_checkpoint(checkpoint_json: str | Path):
    _require_tf()
    ckpt = json.loads(Path(checkpoint_json).read_text(encoding="utf-8"))
    model_path = Path(ckpt["model_path"])
    if not model_path.is_absolute():
        model_path = (Path(checkpoint_json).parent / model_path).resolve()
    model = _load_tf_model_file(model_path)
    tokenizer = BPETokenizer.from_dict(ckpt["tokenizer"])
    return ckpt, model, tokenizer


def evaluate_tensorflow(
    checkpoint_json: str | Path,
    split_path: str | Path,
    *,
    batch_size: int = 64,
    threshold: float | None = None,
    output_path: str | Path | None = None,
) -> dict:
    _require_tf()
    ckpt, model, _ = _load_tf_checkpoint(checkpoint_json)
    ids, mask, labels = load_npz_split(split_path)
    logits = model.predict(
        {"input_ids": ids.astype(np.int32), "attention_mask": mask.astype(np.int32)},
        batch_size=batch_size,
        verbose=0,
    )
    probs = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
    thr = float(ckpt.get("best_threshold", 0.5) if threshold is None else threshold)
    metrics = binary_metrics(labels, probs, threshold=thr)
    metrics["threshold"] = thr
    metrics["backend"] = "tensorflow"
    metrics["checkpoint"] = str(checkpoint_json)
    metrics["split_path"] = str(split_path)
    if output_path:
        save_metrics(metrics, output_path)
    return metrics


def predict_tensorflow(
    checkpoint_json: str | Path,
    texts: Iterable[str],
    *,
    threshold: float | None = None,
    batch_size: int = 64,
) -> list[dict]:
    _require_tf()
    ckpt, model, tokenizer = _load_tf_checkpoint(checkpoint_json)
    max_len = int(ckpt["model_config"]["max_len"])
    thr = float(ckpt.get("best_threshold", 0.5) if threshold is None else threshold)

    text_list = list(texts)
    ids_rows = []
    mask_rows = []
    for text in text_list:
        ids = tokenizer.encode(text, max_len=max_len)
        mask = [1] * len(ids)
        if len(ids) < max_len:
            pad = [tokenizer.pad_id] * (max_len - len(ids))
            ids = ids + pad
            mask = mask + [0] * len(pad)
        ids_rows.append(ids)
        mask_rows.append(mask)

    ids = np.asarray(ids_rows, dtype=np.int32)
    mask = np.asarray(mask_rows, dtype=np.int32)
    logits = model.predict({"input_ids": ids, "attention_mask": mask}, batch_size=batch_size, verbose=0)
    probs = (1.0 / (1.0 + np.exp(-logits.reshape(-1)))).tolist()

    out = []
    for text, prob in zip(text_list, probs):
        out.append(
            {
                "text": text,
                "prob_ai": float(prob),
                "label_pred": int(prob >= thr),
                "threshold": thr,
                "backend": "tensorflow",
            }
        )
    return out
