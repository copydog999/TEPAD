"""La Réunion dataset loader (cross-dataset validation; real, full shading, 7 shading codes).

Data source
-----------
Zenodo ``10.5281/zenodo.7828879`` (Data in Brief 10.1016/j.dib.2022.108723):
a 4 kW residential rooftop plant (University of La Réunion).

* ``dt1_solar_met.csv``  —— irradiance/meteorology (``GTI``, ``TPV``)
* ``dt2_inverter1_faults.csv`` —— inverter 1 (faulty, ``Pg``, ``Fault``)
* ``dt3_inverter2_normal.csv`` —— inverter 2 (healthy, ``Pg``, used as the **twin reference**)
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from ..config import LAREUNION_TIME_WINDOW, dataset_raw_dir

DT1 = "dt1_solar_met.csv"
DT2 = "dt2_inverter1_faults.csv"
DT3 = "dt3_inverter2_normal.csv"


def _load_timeseries(path) -> pd.DataFrame:
    """Read and index a time series (raw read before 1-minute resampling)."""
    d = pd.read_csv(path, on_bad_lines="skip")
    d["time"] = pd.to_datetime(d["time"].astype(str), errors="coerce", utc=True)
    return d.dropna(subset=["time"]).set_index("time")


def load() -> Tuple[pd.DataFrame, dict]:
    """Load and align dt1/dt2/dt3, returning a merged table at 1-minute resolution.

    Returns
    -------
    (df, spec) : Tuple[pandas.DataFrame, dict]
        ``df`` columns: ``time, GTI, TPV, P1, Fault, P2, _site``.
    """
    root = dataset_raw_dir("lareunion")
    g1 = _load_timeseries(root / DT1)[["GTI", "TPV"]].resample("1min").mean()
    g2 = _load_timeseries(root / DT2)[["Pg", "Fault"]].resample("1min").agg({"Pg": "mean", "Fault": "max"})
    g3 = _load_timeseries(root / DT3)[["Pg"]].resample("1min").mean()

    merged = g1.join(g2, how="inner").join(g3, how="inner", rsuffix="_2").dropna().reset_index()
    merged.columns = ["time", "GTI", "TPV", "P1", "Fault", "P2"]

    # Restrict to the analysis window so that downloading a longer irradiance
    # series from Zenodo yields the same days as the reported run.
    start, end = LAREUNION_TIME_WINDOW
    merged = merged[(merged["time"] >= pd.Timestamp(start))
                    & (merged["time"] <= pd.Timestamp(end))].reset_index(drop=True)
    merged["_site"] = "plant"

    spec = dict(site_col="_site",
                g_col="GTI",
                p_col="P1",
                pref_col="P2",                 # healthy twin inverter
                label_col="Fault",
                normal_value=0.0,
                time_col="time",
                date_col="time",
                pref_source_hint="twin_inverter")
    return merged, spec


if __name__ == "__main__":  # pragma: no cover - minimal example
    df, spec = load()
    print("rows:", len(df), "| cols:", list(df.columns))
