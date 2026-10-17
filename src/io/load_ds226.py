"""DS226 dataset loader (cross-dataset validation; real, 5 classes, about 44 daily curves).

Data source
-----------
PV-Fault-DS226 (Zenodo ``10.5281/zenodo.19259504``): a two-panel test bench in
Douala, Cameroon, with 226,175 sub-second synchronized observations
(V1/V2, I1/I2, T1/T2, Irr) and 5 HIL fault classes
(Normal / Short Circuit / Open Circuit / Degradation / Shading).
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from ..config import dataset_raw_dir

FILE = "PV_Diagnostic_Final_Labels.csv"


def load() -> Tuple[pd.DataFrame, dict]:
    """Load DS226 (including the label file).

    Returns
    -------
    (df, spec) : Tuple[pandas.DataFrame, dict]
    """
    f = dataset_raw_dir("ds226") / FILE
    df = pd.read_csv(f, sep=";", low_memory=False)
    spec = dict(site_col="Label_Num_S1",           # consistent with the existing implementation (each day's subset shares one label)
                g_col="Irr",
                p_col="P1",
                pref_col=None,
                label_col="Label_S1",
                normal_value="Normal",
                time_col="timestamp",
                date_col="timestamp")
    return df, spec


if __name__ == "__main__":  # pragma: no cover - minimal example
    df, spec = load()
    print("rows:", len(df), "| cols:", list(df.columns))
    print("labels:", df["Label_S1"].value_counts().to_dict())
