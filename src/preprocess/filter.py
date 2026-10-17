"""Filtering utilities: daylight mask, over-long group rejection, exact deduplication."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..config import DAYLIGHT_QUANTILE, MAX_ROWS_PER_DAY, MIN_DAY_POINTS


def daylight_threshold(g: np.ndarray, quantile: float = DAYLIGHT_QUANTILE) -> float:
    """Daylight irradiance threshold: the ``quantile`` quantile of ``G`` (not below 0)."""
    return max(0.0, float(np.nanquantile(np.asarray(g, dtype=float), quantile)))


def enough_points(n: int, min_points: int = MIN_DAY_POINTS) -> bool:
    """Whether a daily curve has enough valid points."""
    return n >= min_points


def drop_long_groups(df: pd.DataFrame, key_cols: Sequence[str],
                     max_rows: int = MAX_ROWS_PER_DAY) -> pd.DataFrame:
    """Drop (key) groups with more than ``max_rows`` rows (common with duplicated acquisition/merge errors)."""
    sizes = df.groupby(list(key_cols)).size()
    excl = sizes[sizes > max_rows]
    if len(excl) == 0:
        return df
    idx = df.set_index(list(key_cols)).index
    return df[~idx.isin(excl.index)]


def drop_exact_duplicates(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Drop exact duplicates on the given columns."""
    return df.drop_duplicates(subset=list(cols))


if __name__ == "__main__":  # pragma: no cover - minimal example
    gg = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    print("thr =", daylight_threshold(gg), "| enough =", enough_points(12))
