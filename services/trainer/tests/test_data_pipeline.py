import pandas as pd

from src.data_pipeline import load_and_split


def test_load_and_split():
    split = load_and_split("../../train_data/balanced_ai_human_prompts.csv")
    assert len(split.train) > 0
    assert len(split.val) > 0
    assert len(split.test) > 0

