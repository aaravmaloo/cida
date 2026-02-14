from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from cida.tokenizer import BPETokenizer


@dataclass
class DatasetSplit:
    texts: list[str]
    labels: list[int]


_WORKER_TOKENIZER: BPETokenizer | None = None
_WORKER_MAX_LEN: int = 0


def _init_encode_worker(tokenizer_payload: dict, max_len: int) -> None:
    global _WORKER_TOKENIZER, _WORKER_MAX_LEN
    _WORKER_TOKENIZER = BPETokenizer.from_dict(tokenizer_payload)
    _WORKER_MAX_LEN = max_len


def _encode_text_worker(text: str) -> tuple[np.ndarray, np.ndarray]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("Tokenizer worker was not initialized.")
    ids = _WORKER_TOKENIZER.encode(text, max_len=_WORKER_MAX_LEN)
    seq_len = min(len(ids), _WORKER_MAX_LEN)

    row_ids = np.full((_WORKER_MAX_LEN,), _WORKER_TOKENIZER.pad_id, dtype=np.int32)
    row_mask = np.zeros((_WORKER_MAX_LEN,), dtype=np.int8)
    if seq_len:
        row_ids[:seq_len] = np.asarray(ids[:seq_len], dtype=np.int32)
        row_mask[:seq_len] = 1
    return row_ids, row_mask


def read_labeled_csv(csv_path: str | Path, text_col: str, label_col: str) -> DatasetSplit:
    texts: list[str] = []
    labels: list[int] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {text_col, label_col} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing column(s) in CSV: {sorted(missing)}")

        for row in reader:
            text = (row.get(text_col) or "").strip()
            raw_label = (row.get(label_col) or "").strip()
            if not text:
                continue
            if raw_label not in {"0", "1"}:
                raise ValueError(
                    f"Invalid label '{raw_label}'. Expected 0/1 in column '{label_col}'."
                )
            texts.append(text)
            labels.append(int(raw_label))

    if not texts:
        raise ValueError("No usable rows found in dataset.")
    return DatasetSplit(texts=texts, labels=labels)


def stratified_split(
    texts: list[str],
    labels: list[int],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, DatasetSplit]:
    if len(texts) != len(labels):
        raise ValueError("texts and labels length mismatch.")
    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0 and (val_ratio + test_ratio) < 1.0):
        raise ValueError("Invalid val/test ratios. Must satisfy val + test < 1.")

    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {0: [], 1: []}
    for idx, label in enumerate(labels):
        by_class[label].append(idx)

    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for label, idxs in by_class.items():
        if not idxs:
            continue
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        if n_test + n_val >= n:
            overflow = (n_test + n_val) - (n - 1)
            if overflow > 0:
                n_val = max(0, n_val - overflow)

        test_idxs = idxs[:n_test]
        val_idxs = idxs[n_test : n_test + n_val]
        train_idxs = idxs[n_test + n_val :]

        split_indices["train"].extend(train_idxs)
        split_indices["val"].extend(val_idxs)
        split_indices["test"].extend(test_idxs)

    for key in split_indices:
        rng.shuffle(split_indices[key])

    out: dict[str, DatasetSplit] = {}
    for split_name, idxs in split_indices.items():
        out[split_name] = DatasetSplit(
            texts=[texts[i] for i in idxs],
            labels=[labels[i] for i in idxs],
        )
    return out


def encode_split(
    tokenizer: BPETokenizer,
    split: DatasetSplit,
    max_len: int,
    workers: int = 1,
    chunk_size: int | None = None,
    pool: ProcessPoolExecutor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(split.texts)
    input_ids = np.full((n, max_len), tokenizer.pad_id, dtype=np.int32)
    attention_mask = np.zeros((n, max_len), dtype=np.int8)
    labels = np.array(split.labels, dtype=np.float32)

    if workers <= 1 or n < 64:
        for i, text in enumerate(split.texts):
            ids = tokenizer.encode(text, max_len=max_len)
            seq_len = min(len(ids), max_len)
            input_ids[i, :seq_len] = np.asarray(ids[:seq_len], dtype=np.int32)
            attention_mask[i, :seq_len] = 1
        return input_ids, attention_mask, labels

    effective_workers = max(1, min(workers, n))
    if chunk_size is None:
        chunk_size = max(8, n // (effective_workers * 8))

    if pool is not None:
        for i, (row_ids, row_mask) in enumerate(pool.map(_encode_text_worker, split.texts, chunksize=chunk_size)):
            input_ids[i] = row_ids
            attention_mask[i] = row_mask
        return input_ids, attention_mask, labels

    spawn_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=effective_workers,
        mp_context=spawn_ctx,
        initializer=_init_encode_worker,
        initargs=(tokenizer.to_dict(), max_len),
    ) as local_pool:
        for i, (row_ids, row_mask) in enumerate(
            local_pool.map(_encode_text_worker, split.texts, chunksize=chunk_size)
        ):
            input_ids[i] = row_ids
            attention_mask[i] = row_mask
    return input_ids, attention_mask, labels


def save_npz_split(
    file_path: str | Path,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    labels: np.ndarray,
    compressed: bool = True,
) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    else:
        np.savez(path, input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def load_npz_split(file_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(file_path)
    return data["input_ids"], data["attention_mask"], data["labels"]


def class_counts(labels: Iterable[int]) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for label in labels:
        counts[int(label)] = counts.get(int(label), 0) + 1
    return counts


def prepare_dataset(
    input_csv: str | Path,
    output_dir: str | Path,
    text_col: str = "text",
    label_col: str = "generated",
    vocab_size: int = 8000,
    min_pair_freq: int = 2,
    max_len: int = 512,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    lowercase: bool = True,
    workers: int = 0,
    chunk_size: int | None = None,
    compression: str = "none",
) -> dict:
    if compression not in {"none", "compressed"}:
        raise ValueError("compression must be one of: none, compressed.")
    if workers < 0:
        raise ValueError("workers must be >= 0 (0 means use all CPU cores).")
    cpu_total = os.cpu_count() or 1
    effective_workers = cpu_total if workers == 0 else workers

    source = read_labeled_csv(input_csv, text_col=text_col, label_col=label_col)
    splits = stratified_split(
        source.texts,
        source.labels,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    tokenizer = BPETokenizer(lowercase=lowercase)
    tokenizer.train(
        texts=splits["train"].texts,
        vocab_size=vocab_size,
        min_pair_freq=min_pair_freq,
    )

    out_root = Path(output_dir)
    tokenizer_path = out_root / "tokenizer" / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    data_dir = out_root / "data"
    if effective_workers > 1:
        spawn_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=spawn_ctx,
            initializer=_init_encode_worker,
            initargs=(tokenizer.to_dict(), max_len),
        ) as shared_pool:
            for split_name in ("train", "val", "test"):
                ids, mask, labels = encode_split(
                    tokenizer,
                    splits[split_name],
                    max_len=max_len,
                    workers=effective_workers,
                    chunk_size=chunk_size,
                    pool=shared_pool,
                )
                save_npz_split(
                    data_dir / f"{split_name}.npz",
                    ids,
                    mask,
                    labels,
                    compressed=(compression == "compressed"),
                )
    else:
        for split_name in ("train", "val", "test"):
            ids, mask, labels = encode_split(
                tokenizer,
                splits[split_name],
                max_len=max_len,
                workers=1,
                chunk_size=chunk_size,
            )
            save_npz_split(
                data_dir / f"{split_name}.npz",
                ids,
                mask,
                labels,
                compressed=(compression == "compressed"),
            )

    metadata = {
        "input_csv": str(input_csv),
        "text_col": text_col,
        "label_col": label_col,
        "label_mapping": {"0": "human", "1": "ai"},
        "max_len": max_len,
        "vocab_size": tokenizer.vocab_size,
        "requested_vocab_size": vocab_size,
        "min_pair_freq": min_pair_freq,
        "lowercase": lowercase,
        "workers": effective_workers,
        "chunk_size": chunk_size,
        "compression": compression,
        "seed": seed,
        "splits": {
            "train_size": len(splits["train"].labels),
            "val_size": len(splits["val"].labels),
            "test_size": len(splits["test"].labels),
            "train_class_counts": class_counts(splits["train"].labels),
            "val_class_counts": class_counts(splits["val"].labels),
            "test_class_counts": class_counts(splits["test"].labels),
        },
        "artifacts": {
            "tokenizer_path": str(tokenizer_path),
            "data_dir": str(data_dir),
        },
    }

    (out_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return metadata
