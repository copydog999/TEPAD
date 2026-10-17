"""Daily splitting: organize pointwise data into "daily curve" records (TEPAD's unit of analysis)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import DAYLIGHT_QUANTILE, MIN_DAY_POINTS
from .filter import daylight_threshold


@dataclass
class DailyCurve:
    """One daily curve and its label.

    Attributes
    ----------
    dataset : str
        Dataset name.
    site : str
        Site / device identifier.
    date : str
        Date (ISO string).
    label : int
        0 = normal, >0 = abnormal (always 0 for datasets without labels).
    cls : str
        Original class name (used for per-class recall).
    g, p : np.ndarray
        Normalized irradiance and normalized power.
    pref : np.ndarray or None
        Normalized external reference power (``None`` if unavailable).
    n_obs : int
        Number of valid points.
    extra : dict
        Additional fields (reserved for extension).
    """

    dataset: str
    site: str
    date: str
    label: int
    cls: str
    g: np.ndarray
    p: np.ndarray
    pref: Optional[np.ndarray] = None
    n_obs: int = 0
    extra: dict = field(default_factory=dict)


def build_curves(df: pd.DataFrame, dataset: str, site_col: str, g_col: str, p_col: str,
                 pref_col: Optional[str] = None, label_col: Optional[str] = None,
                 normal_value=None, time_col: Optional[str] = None,
                 date_col: str = "date", min_points: int = MIN_DAY_POINTS,
                 pref_source_hint: Optional[str] = None,
                 group_extra_col: Optional[str] = None,
                 class_names: Optional[List[str]] = None) -> List[DailyCurve]:
    """Split a pointwise table into a list of daily curves.

    Flow: date parsing → global min-max normalization (bounds estimated inside the
    daylight mask) → grouping by ``(site, date)`` → daylight filtering (``G > 5%``
    quantile) → minimum point count filtering → labeling.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw (already cleaned) pointwise data.
    dataset : str
        Dataset name.
    site_col, g_col, p_col : str
        Site, irradiance, and power column names.
    pref_col : str, optional
        External reference power column name (e.g. ``modeled_power``).
    label_col : str, optional
        Label column name; ``None`` means unlabeled (deployment feasibility mode).
    normal_value : optional
        Value of the normal class; if ``None``, the maximum of ``label_col`` is used
        as the abnormality indicator.
    time_col : str, optional
        Time column name; uses ``date_col`` when ``None``.
    date_col : str
        Name of an existing date column (used when ``time_col`` is ``None``).
    min_points : int
        Minimum number of valid points per daily curve.
    pref_source_hint : str, optional
        Hint about the external reference source (``"twin_inverter"`` etc.).

    Returns
    -------
    List[DailyCurve]
    """
    df = df.copy()
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        df["_date"] = df[time_col].dt.date
    else:
        df["_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        df = df.dropna(subset=["_date"])
    df = df.dropna(subset=[g_col, p_col])

    g_raw = df[g_col].to_numpy(dtype=float)
    p_raw = df[p_col].to_numpy(dtype=float)
    thr = daylight_threshold(g_raw, DAYLIGHT_QUANTILE)
    mask = g_raw > thr
    if not np.any(mask):
        return []
    gmin, gmax = float(np.nanmin(g_raw[mask])), float(np.nanmax(g_raw[mask]))
    pmin, pmax = float(np.nanmin(p_raw[mask])), float(np.nanmax(p_raw[mask]))
    df["_g"] = np.clip((g_raw - gmin) / (gmax - gmin + 1e-12), 0.0, 1.0)
    df["_p"] = np.clip((p_raw - pmin) / (pmax - pmin + 1e-12), 0.0, 1.0)
    has_pref = pref_col is not None and pref_col in df.columns
    if has_pref:
        pr_raw = df[pref_col].to_numpy(dtype=float)
        df["_pr"] = np.clip((pr_raw - pmin) / (pmax - pmin + 1e-12), 0.0, 1.0)
    curves: List[DailyCurve] = []
    keys = [site_col, "_date"] + ([group_extra_col] if group_extra_col else [])
    for key, grp in df.groupby(keys):
        site, date = key[0], key[1]
        m = grp[g_col].to_numpy(dtype=float) > thr
        g = grp["_g"].to_numpy(dtype=float)[m]
        p = grp["_p"].to_numpy(dtype=float)[m]
        if len(g) < min_points:
            continue
        pref = grp["_pr"].to_numpy(dtype=float)[m] if has_pref else None
        label, cls = 0, ""
        if label_col is not None:
            vals = grp[label_col]
            if normal_value is None:
                label = int(np.nanmax(vals.astype(float)))
                cls = str(vals.mode().iloc[0])
            else:
                abnormal = vals[vals != normal_value]
                label = int(len(abnormal) > 0)
                cls = str(abnormal.mode().iloc[0]) if label == 1 else str(normal_value)
            if class_names is not None:
                try:
                    cls = str(class_names[int(label)])
                except (IndexError, ValueError, TypeError):
                    pass
        curves.append(DailyCurve(dataset=dataset, site=str(site), date=str(date),
                                 label=label, cls=cls, g=g, p=p, pref=pref,
                                 n_obs=len(g), extra={"pref_source_hint": pref_source_hint}))
    return curves


if __name__ == "__main__":  # pragma: no cover - minimal example
    demo = pd.DataFrame({
        "site": ["A"] * 40, "ts": pd.date_range("2024-01-01", periods=40, freq="10min"),
        "G": np.linspace(0, 900, 40), "P": np.linspace(0, 800, 40), "lab": ["Normal"] * 40,
    })
    cs = build_curves(demo, "demo", "site", "G", "P", label_col="lab", normal_value="Normal",
                      time_col="ts")
    print("curves:", len(cs), "| first n_obs:", cs[0].n_obs if cs else None)
