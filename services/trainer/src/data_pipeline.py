from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

_TEXT_WS = re.compile(r"\s+")


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def normalize_text(text: str) -> str:
    return _TEXT_WS.sub(" ", text.strip().lower())


def simhash(text: str, bits: int = 64) -> int:
    tokens = normalize_text(text).split()
    if not tokens:
        return 0

    v = [0] * bits
    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1

    fingerprint = 0
    for i, score in enumerate(v):
        if score >= 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dedupe_simhash(df: pd.DataFrame, text_col: str = "text", threshold: int = 3) -> pd.DataFrame:
    buckets: dict[int, list[int]] = {}
    keep_mask = []

    for text in df[text_col].tolist():
        fp = simhash(text)
        prefix = fp >> 48
        neighbors = buckets.get(prefix, [])

        duplicate = False
        for other in neighbors:
            if hamming(fp, other) <= threshold:
                duplicate = True
                break

        if duplicate:
            keep_mask.append(False)
            continue

        buckets.setdefault(prefix, []).append(fp)
        keep_mask.append(True)

    return df.loc[keep_mask].reset_index(drop=True)


def load_and_split(
    csv_path: str,
    text_col: str = "text",
    label_col: str = "generated",
    seed: int = 42,
) -> SplitData:
    df = pd.read_csv(csv_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns {text_col!r} and {label_col!r}")

    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(-1).astype(int)
    df = df[(df[text_col].str.strip() != "") & (df[label_col].isin([0, 1]))].reset_index(drop=True)

    df = dedupe_simhash(df, text_col=text_col, threshold=3)

    train_val, test = train_test_split(
        df,
        test_size=0.1,
        random_state=seed,
        stratify=df[label_col],
    )
    train, val = train_test_split(
        train_val,
        test_size=0.111111,
        random_state=seed,
        stratify=train_val[label_col],
    )

    return SplitData(train=train.reset_index(drop=True), val=val.reset_index(drop=True), test=test.reset_index(drop=True))

