"""Per-class recall statistics."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def class_names(curves) -> List[str]:
    """Return the class names in order of appearance."""
    seen: List[str] = []
    for c in curves:
        if c.cls not in seen:
            seen.append(c.cls)
    return seen


def per_class_recall(curves, flags: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute the recall and sample count of each class.

    Parameters
    ----------
    curves : List[DailyCurve]
        Daily curves (supplying ``cls``).
    flags : np.ndarray
        Decision flags (the level actually used, usually the cascade).

    Returns
    -------
    dict
        ``{class: {"recall": float, "n": int}}``.
    """
    flags = np.asarray(flags, dtype=bool)
    groups: Dict[str, List[bool]] = {}
    for c, f in zip(curves, flags):
        groups.setdefault(c.cls, []).append(bool(f))
    return {k: {"recall": float(np.mean(v)), "n": int(len(v))} for k, v in groups.items()}


def per_class_table(curves, s1: np.ndarray, s2: np.ndarray, casc: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Return per-class recall for the "stage 1 / stage 2 / cascade" levels (used in paper Table 3)."""
    out: Dict[str, Dict[str, float]] = {}
    for level, name in ((s1, "stage1"), (s2, "stage2"), (casc, "cascade")):
        rec = per_class_recall(curves, level)
        for k, v in rec.items():
            out.setdefault(k, {})[name] = v["recall"]
            out[k]["n"] = v["n"]
    return out


if __name__ == "__main__":  # pragma: no cover - minimal example
    from ..preprocess.daily_split import DailyCurve
    cs = [DailyCurve("d", "s", "1", 0, "normal", np.zeros(3), np.zeros(3)),
          DailyCurve("d", "s", "2", 1, "shading", np.zeros(3), np.zeros(3))]
    print(per_class_recall(cs, np.array([False, True])))
