from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from cida.tf_backend import _load_tf_model_file
from cida.tokenizer import BPETokenizer


def predict_h5(
    model_h5_path: str | Path,
    tokenizer_path: str | Path,
    texts: Iterable[str],
    *,
    max_len: int,
    threshold: float = 0.5,
    batch_size: int = 64,
) -> list[dict]:
    model = _load_tf_model_file(model_h5_path)
    tokenizer = BPETokenizer.load(tokenizer_path)

    text_list = list(texts)
    if not text_list:
        return []

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
                "label_pred": int(prob >= threshold),
                "threshold": float(threshold),
            }
        )
    return out


def predict_from_bundle(
    bundle_config_path: str | Path,
    texts: Iterable[str],
    *,
    threshold: float | None = None,
    batch_size: int = 64,
) -> list[dict]:
    bundle_path = Path(bundle_config_path)
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))

    model_path = Path(payload["model_path"])
    tokenizer_path = Path(payload["tokenizer_path"])
    if not model_path.is_absolute():
        model_path = (bundle_path.parent / model_path).resolve()
    if not tokenizer_path.is_absolute():
        tokenizer_path = (bundle_path.parent / tokenizer_path).resolve()

    max_len = int(payload["max_len"])
    thr = float(payload["threshold"] if threshold is None else threshold)
    return predict_h5(
        model_h5_path=model_path,
        tokenizer_path=tokenizer_path,
        texts=texts,
        max_len=max_len,
        threshold=thr,
        batch_size=batch_size,
    )

