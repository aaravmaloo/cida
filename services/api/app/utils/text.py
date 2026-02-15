import re

_WORD_RE = re.compile(r"\b[\w']+\b", flags=re.UNICODE)
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def sentences(text: str) -> list[str]:
    items = [chunk.strip() for chunk in _SENTENCE_RE.findall(text) if chunk.strip()]
    return items if items else [text.strip()] if text.strip() else []


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

