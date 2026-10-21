"""Label 100 complaints yourself to estimate the label-noise ceiling.

    python -m scripts.review_sample sample   # writes data/review/to_label.csv (blind) and key.csv
    # fill the my_label column in to_label.csv with one of the 7 labels
    python -m scripts.review_sample score    # your agreement with the dataset labels

Sampled from train, so the test set stays untouched. If you agree with the
dataset only ~85% of the time, a model near 0.83 is close to the ceiling.
"""

import sys

import pandas as pd
from sklearn.metrics import classification_report

from src import config
from src.data import load_data
from src.labels import LABELS

DIR = config.DATA_DIR / "review"


def sample(n_per_class=15, seed=7):
    train, _ = load_data()
    picked = (train.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(n_per_class, random_state=seed))
              .sample(frac=1, random_state=seed)  # shuffle so classes are not in blocks
              .head(100))
    DIR.mkdir(parents=True, exist_ok=True)
    picked = picked.reset_index().rename(columns={"index": "train_row"})
    picked[["train_row", "text"]].assign(my_label="").to_csv(DIR / "to_label.csv", index=False)
    picked[["train_row", "label"]].to_csv(DIR / "key.csv", index=False)
    print(f"wrote {len(picked)} complaints to {DIR / 'to_label.csv'}; labels: {', '.join(LABELS)}")


def score():
    mine = pd.read_csv(DIR / "to_label.csv").dropna(subset=["my_label"])
    df = mine.merge(pd.read_csv(DIR / "key.csv"), on="train_row")
    bad = set(df["my_label"]) - set(LABELS)
    if bad:
        raise SystemExit(f"unknown labels in my_label: {bad}")
    print(f"labelled {len(df)}; agreement with dataset: {(df['my_label'] == df['label']).mean():.2f}")
    print(classification_report(df["label"], df["my_label"], zero_division=0))
    print(df[df["my_label"] != df["label"]][["train_row", "label", "my_label"]].to_string(index=False))


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "sample"]()
