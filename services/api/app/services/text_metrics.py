from __future__ import annotations

import math
from dataclasses import dataclass

from app.utils.text import clamp, sentences, words


@dataclass
class TextMetrics:
    readability: dict[str, float]
    complexity: float
    burstiness: float
    vocab_diversity: float
    word_count: int
    estimated_read_time: float


def _syllables(word: str) -> int:
    vowels = "aeiouy"
    w = word.lower().strip()
    if not w:
        return 1
    count = 0
    prev = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev:
            count += 1
        prev = is_vowel
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _mtld(tokens: list[str], threshold: float = 0.72) -> float:
    if not tokens:
        return 0.0
    factors = 0
    segment_start = 0
    seen: set[str] = set()

    for idx, tok in enumerate(tokens, start=1):
        seen.add(tok)
        ttr = len(seen) / max(1, idx - segment_start)
        if ttr <= threshold:
            factors += 1
            segment_start = idx
            seen = set()

    remainder = len(tokens) - segment_start
    if remainder > 0 and seen:
        ttr = len(seen) / remainder
        factors += (1 - ttr) / max(1e-6, 1 - threshold)

    return len(tokens) / max(1e-6, factors)


def compute_metrics(text: str) -> TextMetrics:
    sent = sentences(text)
    toks = words(text)

    wc = len(toks)
    sc = max(1, len(sent))
    syllable_count = sum(_syllables(token) for token in toks)

    asl = wc / sc
    asw = syllable_count / max(1, wc)

    flesch = 206.835 - (1.015 * asl) - (84.6 * asw)
    grade = (0.39 * asl) + (11.8 * asw) - 15.59
    complex_words = sum(1 for token in toks if _syllables(token) >= 3)
    smog = 1.0430 * math.sqrt(complex_words * (30 / sc)) + 3.1291 if sc else 0.0

    sentence_lengths = [len(words(s)) for s in sent if s.strip()]
    mean_len = sum(sentence_lengths) / max(1, len(sentence_lengths))
    variance = (
        sum((x - mean_len) ** 2 for x in sentence_lengths) / max(1, len(sentence_lengths))
        if sentence_lengths
        else 0.0
    )
    std_dev = math.sqrt(variance)
    burstiness = clamp(std_dev / max(1.0, mean_len), 0.0, 1.0)

    punctuation_count = sum(1 for c in text if c in ",;:!?-")
    punctuation_entropy = clamp(punctuation_count / max(1, len(text)), 0.0, 0.2)
    clause_density_proxy = clamp(text.count(",") / max(1, sc), 0.0, 4.0)
    complexity = clamp((mean_len / 30) * 0.55 + (clause_density_proxy / 4) * 0.25 + (punctuation_entropy / 0.2) * 0.2, 0.0, 1.0)

    unique = len(set(toks))
    ttr = unique / max(1, wc)
    mtld_score = _mtld(toks)
    vocab_diversity = clamp((ttr * 0.55) + (min(mtld_score, 120) / 120 * 0.45), 0.0, 1.0)

    read_minutes = wc / 230.0

    return TextMetrics(
        readability={
            "flesch_reading_ease": round(clamp(flesch, -20.0, 120.0), 3),
            "flesch_kincaid_grade": round(clamp(grade, -3.0, 18.0), 3),
            "smog_index": round(clamp(smog, 0.0, 18.0), 3),
        },
        complexity=round(complexity, 4),
        burstiness=round(burstiness, 4),
        vocab_diversity=round(vocab_diversity, 4),
        word_count=wc,
        estimated_read_time=round(read_minutes, 2),
    )

