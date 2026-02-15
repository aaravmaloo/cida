import pandas as pd

from src.data_pipeline import load_and_split


def test_load_and_split_multi_source_auto_label(tmp_path):
    rows = []
    for i in range(40):
        rows.append({"text": f"human sample {i}", "label": 0})
        rows.append({"text": f"ai sample {i}", "label": 1})
    csv_a = tmp_path / "train_a.csv"
    pd.DataFrame(rows).to_csv(csv_a, index=False)

    rows_b = []
    for i in range(20):
        rows_b.append({"text": f"human generated sample {i}", "generated": 0})
        rows_b.append({"text": f"ai generated sample {i}", "generated": 1})
    csv_b = tmp_path / "train_b.csv"
    pd.DataFrame(rows_b).to_csv(csv_b, index=False)

    split = load_and_split([str(csv_a), str(csv_b)], label_col="auto")

    assert split.label_col == "label"
    assert len(split.train) > 0
    assert len(split.val) > 0
    assert len(split.test) > 0
    assert set(split.train[split.label_col].unique()).issubset({0, 1})


def test_load_and_split_unlabeled_source_with_default_label(tmp_path):
    labeled_rows = []
    for i in range(30):
        labeled_rows.append({"text": f"human sample {i}", "label": 0})
        labeled_rows.append({"text": f"ai sample {i}", "label": 1})
    labeled_csv = tmp_path / "labeled.csv"
    pd.DataFrame(labeled_rows).to_csv(labeled_csv, index=False)

    unlabeled_rows = [{"text": f"synthetic ai sample {i}", "source": "generator"} for i in range(30)]
    unlabeled_csv = tmp_path / "unlabeled.csv"
    pd.DataFrame(unlabeled_rows).to_csv(unlabeled_csv, index=False)

    split = load_and_split(
        [str(labeled_csv), str(unlabeled_csv)],
        label_col="auto",
        unlabeled_default_label=1,
    )

    assert split.label_col == "label"
    assert len(split.train) > 0
    assert set(split.train[split.label_col].unique()).issubset({0, 1})

