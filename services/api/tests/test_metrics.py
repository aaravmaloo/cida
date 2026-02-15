import pytest

from app.services.text_metrics import compute_metrics


def test_compute_metrics_non_empty():
    text = "This is a short sentence. This is another one with a bit more variety."
    metrics = compute_metrics(text)

    assert metrics.word_count > 5
    assert 0.0 <= metrics.complexity <= 1.0
    assert 0.0 <= metrics.burstiness <= 1.0
    assert 0.0 <= metrics.vocab_diversity <= 1.0
    assert "flesch_kincaid_grade" in metrics.readability


def test_compute_metrics_empty_safe():
    metrics = compute_metrics("")
    assert metrics.word_count == 0

