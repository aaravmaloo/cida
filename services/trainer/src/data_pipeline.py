from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

_TEXT_WS = re.compile(r"\s+")
_DEFAULT_AUTO_LABEL_CANDIDATES = ("label", "generated", "target", "is_ai", "ai")


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    text_col: str
    label_col: str


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


def _read_dataset(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                "Parquet loading requires pyarrow or fastparquet. Install dependencies from services/trainer/requirements.txt."
            ) from exc
    raise ValueError(f"Unsupported data file type for {path!r}. Use .csv or .parquet")


def _binary_label_score(series: pd.Series) -> int:
    return int(pd.to_numeric(series, errors="coerce").isin([0, 1]).sum())


def _resolve_label_column(
    df: pd.DataFrame,
    text_col: str,
    requested_label_col: str,
    source_path: str,
    unlabeled_default_label: int | None,
) -> str | None:
    if requested_label_col != "auto":
        if requested_label_col not in df.columns:
            raise ValueError(f"{source_path}: missing requested label column {requested_label_col!r}")
        score = _binary_label_score(df[requested_label_col])
        if score == 0:
            raise ValueError(
                f"{source_path}: label column {requested_label_col!r} has no usable binary labels (expected 0/1)"
            )
        return requested_label_col

    candidates: list[str] = []
    lower_to_name = {str(col).lower(): str(col) for col in df.columns}
    for name in _DEFAULT_AUTO_LABEL_CANDIDATES:
        resolved = lower_to_name.get(name)
        if resolved and resolved != text_col and resolved not in candidates:
            candidates.append(resolved)

    for col in df.columns:
        col_name = str(col)
        lower = col_name.lower()
        if col_name == text_col:
            continue
        if any(token in lower for token in ("label", "target", "class", "generated")) and col_name not in candidates:
            candidates.append(col_name)

    if not candidates:
        if unlabeled_default_label in {0, 1}:
            return None
        raise ValueError(
            f"{source_path}: no candidate label columns found. Pass --label-col explicitly or set --unlabeled-default-label."
        )

    best_col = max(candidates, key=lambda col: _binary_label_score(df[col]))
    if _binary_label_score(df[best_col]) == 0:
        if unlabeled_default_label in {0, 1}:
            return None
        raise ValueError(f"{source_path}: could not infer a binary label column with values 0/1")
    return best_col


def load_and_split(
    data_paths: str | Sequence[str],
    text_col: str = "text",
    label_col: str = "auto",
    seed: int = 42,
    unlabeled_default_label: int | None = None,
) -> SplitData:
    if isinstance(data_paths, (str, Path)):
        sources = [str(data_paths)]
    else:
        sources = [str(path) for path in data_paths]

    if not sources:
        raise ValueError("No dataset paths provided")
    if unlabeled_default_label not in {None, 0, 1}:
        raise ValueError("unlabeled_default_label must be 0, 1, or None")

    canonical_label_col = label_col if label_col != "auto" else "label"
    cleaned_frames: list[pd.DataFrame] = []

    for source in sources:
        df = _read_dataset(source)
        if text_col not in df.columns:
            raise ValueError(f"{source}: missing text column {text_col!r}")

        source_label_col = _resolve_label_column(
            df,
            text_col=text_col,
            requested_label_col=label_col,
            source_path=source,
            unlabeled_default_label=unlabeled_default_label,
        )
        if source_label_col is None:
            frame = df[[text_col]].copy()
            frame[canonical_label_col] = int(unlabeled_default_label)
        else:
            frame = df[[text_col, source_label_col]].copy()
            if source_label_col != canonical_label_col:
                frame = frame.rename(columns={source_label_col: canonical_label_col})

        frame[text_col] = frame[text_col].fillna("").astype(str)
        frame[canonical_label_col] = pd.to_numeric(frame[canonical_label_col], errors="coerce").fillna(-1).astype(int)
        frame = frame[(frame[text_col].str.strip() != "") & (frame[canonical_label_col].isin([0, 1]))].reset_index(
            drop=True
        )
        cleaned_frames.append(frame)

    if not cleaned_frames:
        raise ValueError("No dataset rows available after loading sources")

    df = pd.concat(cleaned_frames, ignore_index=True)
    if df.empty:
        raise ValueError("No valid rows with non-empty text and binary labels (0/1) were found")

    df = dedupe_simhash(df, text_col=text_col, threshold=3)

    train_val, test = train_test_split(
        df,
        test_size=0.1,
        random_state=seed,
        stratify=df[canonical_label_col],
    )
    train, val = train_test_split(
        train_val,
        test_size=0.111111,
        random_state=seed,
        stratify=train_val[canonical_label_col],
    )

    return SplitData(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        text_col=text_col,
        label_col=canonical_label_col,
    )

