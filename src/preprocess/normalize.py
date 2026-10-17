"""Normalization utilities: global min-max and per-device min-max."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

EPS = 1e-12


def minmax(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linearly map an array to ``[0, 1]`` (clipped at the bounds)."""
    return np.clip((np.asarray(arr, dtype=float) - lo) / (hi - lo + EPS), 0.0, 1.0)


def global_bounds(g: np.ndarray, p: np.ndarray,
                  mask: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute ``(gmin, gmax, pmin, pmax)`` inside the daylight mask ``mask``."""
    g = np.asarray(g, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    return (float(np.nanmin(g[mask])), float(np.nanmax(g[mask])),
            float(np.nanmin(p[mask])), float(np.nanmax(p[mask])))


def per_group_minmax(df: pd.DataFrame, cols, group_col: str) -> pd.DataFrame:
    """Apply min-max per group of ``group_col`` (used when devices have different capacities).

    Returns
    -------
    pandas.DataFrame
        A modified copy (the columns are replaced with normalized values).
    """
    out = df.copy()
    for c in cols:
        grp = out.groupby(group_col)[c]
        lo, hi = grp.transform("min"), grp.transform("max")
        out[c] = ((out[c] - lo) / (hi - lo + EPS)).clip(0.0, 1.0)
    return out


if __name__ == "__main__":  # pragma: no cover - minimal example
    print(minmax(np.array([0.0, 5.0, 10.0]), 0.0, 10.0))
