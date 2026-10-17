"""Data preparation for the comparison experiments: the official Ding train/test split.

``build_curves`` needs a global min-max normalization over the whole dataset, so all
data is first split into daily curves once, and the curves are then assigned to the
two splits depending on whether ``(site_num, date)`` appears in the official ``train`` /
``test`` files (the ``(site_num, date)`` keys of the two splits do not overlap).
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from ..config import DING_LABELS, dataset_raw_dir
from ..io import get_loader
from ..preprocess.daily_split import DailyCurve, build_curves

SPLITS: Tuple[str, ...] = ("train", "test")


def split_keys(split: str) -> set:
    """Return the set of ``(site_num, date)`` keys that occur in a given split."""
    root = dataset_raw_dir("ding")
    keys = set()
    for lbl in DING_LABELS:
        f = root / split / "{} {}.csv".format(lbl, split).replace(" ", "_")
        if not f.exists():
            continue
        d = pd.read_csv(f, usecols=["site_num", "date"])
        keys |= set(zip(d["site_num"].astype(str), d["date"].astype(str)))
    return keys


def load_ding_curves() -> List[DailyCurve]:
    """Load all Ding daily curves (identical to the main experiment, 1,331 curves)."""
    df, spec = get_loader("ding")()
    return build_curves(df, "ding", **spec)


def load_ding_split() -> Tuple[List[DailyCurve], List[DailyCurve]]:
    """Return ``(train_curves, test_curves)`` following the official train/test split."""
    curves = load_ding_curves()
    train_keys = split_keys("train")
    test_keys = split_keys("test")
    train = [c for c in curves if (c.site, c.date) in train_keys]
    test = [c for c in curves if (c.site, c.date) in test_keys]
    return train, test


def reference_curves(train: List[DailyCurve]) -> List[DailyCurve]:
    """Fitting set: the normal curves of the training split."""
    return [c for c in train if c.label == 0]


if __name__ == "__main__":  # pragma: no cover - minimal example
    tr, te = load_ding_split()
    ref = reference_curves(tr)
    print("train=%d (normal=%d) | test=%d (normal=%d, abnormal=%d)"
          % (len(tr), len(ref), len(te), sum(c.label == 0 for c in te),
             sum(c.label != 0 for c in te)))
