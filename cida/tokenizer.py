from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class SpecialTokens:
    pad: str = "[PAD]"
    unk: str = "[UNK]"
    cls: str = "[CLS]"
    sep: str = "[SEP]"

    def as_list(self) -> list[str]:
        return [self.pad, self.unk, self.cls, self.sep]


class BPETokenizer:

    def __init__(self, lowercase: bool = True, special_tokens: SpecialTokens | None = None) -> None:
        self.lowercase = lowercase
        self.special = special_tokens or SpecialTokens()
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []
        self.merges: list[tuple[str, str]] = []
        self.merge_ranks: dict[tuple[str, str], int] = {}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.special.pad]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.special.unk]

    @property
    def cls_id(self) -> int:
        return self.token_to_id[self.special.cls]

    @property
    def sep_id(self) -> int:
        return self.token_to_id[self.special.sep]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def _normalize(self, text: str) -> str:
        return text.lower() if self.lowercase else text

    def pretokenize(self, text: str) -> list[str]:
        normalized = self._normalize(text)
        return TOKEN_PATTERN.findall(normalized)

    @staticmethod
    def _count_pairs(words: Counter[tuple[str, ...]]) -> Counter[tuple[str, str]]:
        pair_counts: Counter[tuple[str, str]] = Counter()
        for word_symbols, freq in words.items():
            if len(word_symbols) < 2:
                continue
            for idx in range(len(word_symbols) - 1):
                pair_counts[(word_symbols[idx], word_symbols[idx + 1])] += freq
        return pair_counts

    @staticmethod
    def _merge_pair(
        words: Counter[tuple[str, ...]],
        pair: tuple[str, str],
    ) -> Counter[tuple[str, ...]]:
        merged_symbol = pair[0] + pair[1]
        out: Counter[tuple[str, ...]] = Counter()
        for word_symbols, freq in words.items():
            new_symbols: list[str] = []
            i = 0
            while i < len(word_symbols):
                if (
                    i < len(word_symbols) - 1
                    and word_symbols[i] == pair[0]
                    and word_symbols[i + 1] == pair[1]
                ):
                    new_symbols.append(merged_symbol)
                    i += 2
                else:
                    new_symbols.append(word_symbols[i])
                    i += 1
            out[tuple(new_symbols)] += freq
        return out

    def train(
        self,
        texts: Iterable[str],
        vocab_size: int = 8000,
        min_pair_freq: int = 2,
        max_merges: int = 8000,
    ) -> None:
        if vocab_size < 64:
            raise ValueError("vocab_size must be >= 64 for stable training.")

        word_freq: Counter[str] = Counter()
        for text in texts:
            for token in self.pretokenize(text):
                if token:
                    word_freq[token] += 1

        if not word_freq:
            raise ValueError("Tokenizer training failed: no tokens found in corpus.")

        words: Counter[tuple[str, ...]] = Counter()
        for word, freq in word_freq.items():
            symbols = tuple(word)
            if symbols:
                words[symbols] += freq

        if not words:
            raise ValueError("Tokenizer training failed: no symbol sequences found.")

        symbols: set[str] = set()
        for word_symbols in words:
            symbols.update(word_symbols)

        self.merges = []
        target_non_special = max(1, vocab_size - len(self.special.as_list()))
        merge_budget = max(0, min(max_merges, target_non_special))
        no_new_streak = 0

        for _ in range(merge_budget):
            if len(symbols) >= target_non_special:
                break
            pair_counts = self._count_pairs(words)
            if not pair_counts:
                break
            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < min_pair_freq:
                break
            words = self._merge_pair(words, best_pair)
            merged = best_pair[0] + best_pair[1]
            was_new = merged not in symbols
            symbols.add(merged)
            self.merges.append(best_pair)
            if was_new:
                no_new_streak = 0
            else:
                no_new_streak += 1
            if no_new_streak >= 512:
                break

    
        final_symbol_freq: Counter[str] = Counter()
        for word_symbols, freq in words.items():
            for sym in word_symbols:
                final_symbol_freq[sym] += freq

        if len(final_symbol_freq) > target_non_special:
            selected_symbols = [sym for sym, _ in final_symbol_freq.most_common(target_non_special)]
        else:
            selected_symbols = list(final_symbol_freq.keys())

        merged_outputs = [a + b for (a, b) in self.merges]
        for sym in merged_outputs:
            if sym not in final_symbol_freq and len(selected_symbols) < target_non_special:
                selected_symbols.append(sym)

        specials = self.special.as_list()
    
        selected_symbols = sorted(set(selected_symbols), key=lambda x: (-final_symbol_freq.get(x, 0), x))
        self.id_to_token = specials + selected_symbols
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.id_to_token)}
        self.merge_ranks = {pair: idx for idx, pair in enumerate(self.merges)}

    def _apply_bpe_to_token(self, token: str) -> list[str]:
        if not token:
            return []
        symbols = list(token)
        if len(symbols) == 1:
            return symbols

        while len(symbols) > 1:
            best_pair = None
            best_rank = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank
            if best_pair is None:
                break

            merged_symbol = best_pair[0] + best_pair[1]
            new_symbols: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i + 1] == best_pair[1]:
                    new_symbols.append(merged_symbol)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text: str, max_len: int = 512) -> list[int]:
        if not self.token_to_id:
            raise RuntimeError("Tokenizer is not trained or loaded.")

        ids = [self.cls_id]
        for token in self.pretokenize(text):
            pieces = self._apply_bpe_to_token(token)
            for piece in pieces:
                ids.append(self.token_to_id.get(piece, self.unk_id))

        ids.append(self.sep_id)
        if len(ids) > max_len:
            ids = ids[:max_len]
            ids[-1] = self.sep_id
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        tokens = []
        for idx in ids:
            if 0 <= idx < len(self.id_to_token):
                tok = self.id_to_token[idx]
                if tok in self.special.as_list():
                    continue
                tokens.append(tok)
        return " ".join(tokens)

    def to_dict(self) -> dict:
        return {
            "lowercase": self.lowercase,
            "special_tokens": {
                "pad": self.special.pad,
                "unk": self.special.unk,
                "cls": self.special.cls,
                "sep": self.special.sep,
            },
            "id_to_token": self.id_to_token,
            "merges": [[a, b] for (a, b) in self.merges],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BPETokenizer":
        special = payload.get("special_tokens", {})
        tokenizer = cls(
            lowercase=bool(payload.get("lowercase", True)),
            special_tokens=SpecialTokens(
                pad=special.get("pad", "[PAD]"),
                unk=special.get("unk", "[UNK]"),
                cls=special.get("cls", "[CLS]"),
                sep=special.get("sep", "[SEP]"),
            ),
        )
        tokenizer.id_to_token = list(payload["id_to_token"])
        tokenizer.token_to_id = {tok: idx for idx, tok in enumerate(tokenizer.id_to_token)}
        tokenizer.merges = [(a, b) for a, b in payload.get("merges", [])]
        tokenizer.merge_ranks = {pair: idx for idx, pair in enumerate(tokenizer.merges)}
        return tokenizer

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), ensure_ascii=True, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
