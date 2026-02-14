from __future__ import annotations

import csv
import json
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from types import SimpleNamespace

import numpy as np

from cida.data import load_npz_split
from cida.metrics import best_f1_threshold, binary_metrics, save_metrics
from cida.tokenizer import BPETokenizer

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - runtime dependency gate
    torch = None
    nn = SimpleNamespace(Module=object)
    DataLoader = object
    Dataset = object


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is not installed. Install torch to use --backend torch.")


def _auto_device(device: str | None = None) -> str:
    _require_torch()
    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TorchModelConfig:
    vocab_size: int
    max_len: int
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    ffn_dim: int = 1024
    dropout: float = 0.1
    pooling: str = "cls"
    cls_hidden_dim: int = 256


class NpzDataset(Dataset):
    def __init__(self, input_ids: np.ndarray, attention_mask: np.ndarray, labels: np.ndarray):
        self.input_ids = input_ids.astype(np.int64)
        self.attention_mask = attention_mask.astype(np.int64)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.input_ids[idx]),
            torch.from_numpy(self.attention_mask[idx]),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # attention_mask: [B, T] with 1 for valid tokens, 0 for padding
        key_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        scores = scores.masked_fill(key_mask == 0, -1e9)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        ctx = torch.matmul(probs, v)  # [B, H, T, D]
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.out_proj(ctx)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x), attention_mask))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, cfg: TorchModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    ffn_dim=cfg.ffn_dim,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.cls_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.cls_hidden_dim, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_drop(x)

        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.final_norm(x)

        if self.cfg.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = x[:, 0, :]
        logits = self.classifier(pooled).squeeze(-1)
        return logits


def _linear_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _predict_probs(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for input_ids, attention_mask, y in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(p)
            labels.append(y.numpy())
    return np.concatenate(labels, axis=0), np.concatenate(probs, axis=0)


def train_torch(
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
    grad_clip: float = 1.0,
    label_smoothing: float = 0.0,
    patience: int = 5,
    threshold: float = 0.5,
    seed: int = 42,
    device: str = "auto",
) -> dict:
    _require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    artifact_dir = Path(artifact_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    tokenizer = BPETokenizer.load(artifact_dir / "tokenizer" / "tokenizer.json")

    train_ids, train_mask, train_y = load_npz_split(artifact_dir / "data" / "train.npz")
    val_ids, val_mask, val_y = load_npz_split(artifact_dir / "data" / "val.npz")

    train_loader = DataLoader(
        NpzDataset(train_ids, train_mask, train_y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        NpzDataset(val_ids, val_mask, val_y),
        batch_size=batch_size * 2,
        shuffle=False,
        drop_last=False,
    )

    cfg = TorchModelConfig(
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
    model = TransformerClassifier(cfg)

    target_device = _auto_device(device)
    model.to(target_device)
    use_amp = target_device.startswith("cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_steps = max(1, epochs * len(train_loader))
    scheduler = _linear_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss()

    best_score = -1.0
    best_epoch = -1
    best_threshold = threshold
    best_metrics = None
    wait = 0

    history_rows: list[dict] = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for input_ids, attention_mask, y in train_loader:
            input_ids = input_ids.to(target_device)
            attention_mask = attention_mask.to(target_device)
            y = y.to(target_device)

            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()
            with autocast_ctx:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                if label_smoothing > 0:
                    y_smooth = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
                    loss = criterion(logits, y_smooth)
                else:
                    loss = criterion(logits, y)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = input_ids.size(0)
            loss_sum += loss.item() * bs
            sample_count += bs
            global_step += 1

        train_loss = loss_sum / max(1, sample_count)
        val_true, val_prob = _predict_probs(model, val_loader, target_device)
        val_metrics = binary_metrics(val_true, val_prob, threshold=threshold)
        tuned_threshold, tuned_val_metrics = best_f1_threshold(val_true, val_prob)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "tuned_val_f1": tuned_val_metrics["f1"],
            "tuned_threshold": tuned_threshold,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history_rows.append(row)

        score = tuned_val_metrics["f1"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = tuned_val_metrics
            best_threshold = tuned_threshold
            wait = 0
            checkpoint = {
                "backend": "torch",
                "model_state": model.state_dict(),
                "model_config": asdict(cfg),
                "train_config": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "warmup_steps": warmup_steps,
                    "grad_clip": grad_clip,
                    "label_smoothing": label_smoothing,
                    "patience": patience,
                    "seed": seed,
                },
                "best_epoch": best_epoch,
                "best_threshold": best_threshold,
                "best_val_metrics": best_metrics,
                "metadata": metadata,
                "tokenizer": tokenizer.to_dict(),
            }
            torch.save(checkpoint, out_dir / "best.pt")
        else:
            wait += 1

        if wait >= patience:
            break

    torch.save(
        {
            "backend": "torch",
            "model_state": model.state_dict(),
            "model_config": asdict(cfg),
            "metadata": metadata,
            "tokenizer": tokenizer.to_dict(),
        },
        out_dir / "last.pt",
    )

    history_path = out_dir / "train_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()) if history_rows else [])
        if history_rows:
            writer.writeheader()
            writer.writerows(history_rows)

    train_summary = {
        "backend": "torch",
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "best_val_metrics": best_metrics or {},
        "output_dir": str(out_dir),
        "history_file": str(history_path),
    }
    save_metrics(train_summary, out_dir / "train_summary.json")
    return train_summary


def _load_model_from_checkpoint(checkpoint_path: str | Path, device: str = "auto"):
    _require_torch()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = TorchModelConfig(**ckpt["model_config"])
    model = TransformerClassifier(cfg)
    model.load_state_dict(ckpt["model_state"])
    target_device = _auto_device(device)
    model.to(target_device)
    model.eval()
    tokenizer = BPETokenizer.from_dict(ckpt["tokenizer"])
    return ckpt, model, tokenizer, target_device


def evaluate_torch(
    checkpoint_path: str | Path,
    split_path: str | Path,
    *,
    batch_size: int = 64,
    threshold: float | None = None,
    device: str = "auto",
    output_path: str | Path | None = None,
) -> dict:
    _require_torch()
    ckpt, model, _, target_device = _load_model_from_checkpoint(checkpoint_path, device=device)
    ids, mask, labels = load_npz_split(split_path)
    loader = DataLoader(
        NpzDataset(ids, mask, labels),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    y_true, y_prob = _predict_probs(model, loader, target_device)
    thr = ckpt.get("best_threshold", 0.5) if threshold is None else threshold
    metrics = binary_metrics(y_true, y_prob, threshold=thr)
    metrics["threshold"] = float(thr)
    metrics["backend"] = "torch"
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["split_path"] = str(split_path)
    if output_path:
        save_metrics(metrics, output_path)
    return metrics


def predict_torch(
    checkpoint_path: str | Path,
    texts: Iterable[str],
    *,
    threshold: float | None = None,
    device: str = "auto",
) -> list[dict]:
    _require_torch()
    ckpt, model, tokenizer, target_device = _load_model_from_checkpoint(checkpoint_path, device=device)
    max_len = int(ckpt["model_config"]["max_len"])
    thr = ckpt.get("best_threshold", 0.5) if threshold is None else threshold

    encoded_ids = []
    encoded_mask = []
    text_list = list(texts)
    for text in text_list:
        ids = tokenizer.encode(text, max_len=max_len)
        mask = [1] * len(ids)
        if len(ids) < max_len:
            pad = [tokenizer.pad_id] * (max_len - len(ids))
            ids = ids + pad
            mask = mask + [0] * len(pad)
        encoded_ids.append(ids)
        encoded_mask.append(mask)

    input_ids = torch.tensor(encoded_ids, dtype=torch.long, device=target_device)
    attention_mask = torch.tensor(encoded_mask, dtype=torch.long, device=target_device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()

    out = []
    for text, prob in zip(text_list, probs):
        out.append(
            {
                "text": text,
                "prob_ai": float(prob),
                "label_pred": int(prob >= thr),
                "threshold": float(thr),
                "backend": "torch",
            }
        )
    return out
