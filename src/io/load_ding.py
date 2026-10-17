"""Ding dataset loader (main experiment; synthetic, 7 classes, 1,331 daily curves).

Data source
-----------
The public PV anomaly detection dataset of Ding et al., 30 plants, 7 operating
conditions (normal / spike / inverter_fault / snowy / cloudy / low_value / shading).
Local files live in ``datasets/ding/raw/{train,test}/<label>_<split>.csv``.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from ..config import DING_LABELS, MAX_ROWS_PER_DAY, dataset_raw_dir

SPLITS = ["train", "test"]

#: Feature columns used for exact deduplication (matching Ding's original preprocessing).
FEATURE_COLS = ["instantaneous_output_power", "instantaneous_global_radiation", "clearsky_Tamb",
                "modeled_power", "power_mean", "power_std", "radiation_mean", "radiation_std",
                "simulate_mean", "simulate_std", "tprt_mean", "tprt_std"]

NUMERIC_COLS = ["instantaneous_output_power", "instantaneous_global_radiation", "clearsky_Tamb",
                "modeled_power", "tprt_mean"]

KEY_COLS = ["site_num", "date", "label"]


def load() -> Tuple[pd.DataFrame, dict]:
    """Load and clean the Ding dataset.

    Returns
    -------
    (df, spec) : Tuple[pandas.DataFrame, dict]
        The pointwise table and the column mapping.
    """
    root = dataset_raw_dir("ding")
    frames = []
    for lbl in DING_LABELS:
        for split in SPLITS:
            f = root / split / "{} {}.csv".format(lbl, split).replace(" ", "_")
            d = pd.read_csv(f)
            d.columns = [c.strip() for c in d.columns]
            d = d.rename(columns={d.columns[0]: "row_idx"})
            frames.append(d)
    raw = pd.concat(frames, ignore_index=True)

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"])
    raw["site_num"] = raw["site_num"].astype(int)
    raw["label"] = raw["label"].astype(int)
    for c in NUMERIC_COLS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # Drop abnormally long groups (> MAX_ROWS_PER_DAY rows)
    sizes = raw.groupby(KEY_COLS).size()
    excl = sizes[sizes > MAX_ROWS_PER_DAY]
    if len(excl):
        idx = raw.set_index(KEY_COLS).index
        raw = raw[~idx.isin(excl.index)]
    raw = raw.drop_duplicates(subset=FEATURE_COLS + KEY_COLS)

    spec = dict(site_col="site_num",
                g_col="instantaneous_global_radiation",
                p_col="instantaneous_output_power",
                pref_col="modeled_power",
                label_col="label",
                normal_value=None,          # the label column is already 0..6, with 0=normal
                time_col=None,
                date_col="date",
                group_extra_col="label",    # the same (site,date) appears in several label files
                class_names=DING_LABELS)
    return raw, spec


if __name__ == "__main__":  # pragma: no cover - minimal example
    df, spec = load()
    print("rows:", len(df))
    print("labels:", df["label"].value_counts().sort_index().to_dict())
