# -*- coding: utf-8 -*-
"""Figure 3: per-class box plot of the Dataset A Stage-1 score $d$."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .style import BEST_COLOR, COLORS, legend_below, by_name, new_fig, save

ORDER = ["normal", "spike", "inverter_fault", "snowy", "cloudy", "low_value", "shading"]


def build(results: List[dict], out_dir: Path) -> List[Path]:
    """Generate ``fig3_score_distribution.{pdf,png}``."""
    r = by_name(results)["ding"]
    data, labels = [], []
    if "curves" in r and "det" in r:
        curves, scores = r["curves"], r["det"].d
        values_by_class = {
            label: [scores[i] for i, curve in enumerate(curves) if curve.cls == label]
            for label in ORDER
        }
    else:
        import pandas as pd
        from ..config import DATASETS_DIR
        processed = DATASETS_DIR / "ding" / "processed" / "daily_curves.csv"
        frame = pd.read_csv(processed)
        values_by_class = {
            label: frame.loc[frame["cls"] == label, "stage1_distance"]
            .dropna().astype(float).tolist()
            for label in ORDER
        }
    for c in ORDER:
        vals = values_by_class.get(c, [])
        if vals:
            data.append(vals)
            labels.append(c.replace("_", "\n"))
    fig, ax = new_fig(double=True, height=2.8)
    bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.0),
                    boxprops=dict(lw=0.8), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    medians = [float(np.median(values)) for values in data]
    best_index = int(np.argmax(medians))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(BEST_COLOR if i == best_index else COLORS[i % len(COLORS)])
        patch.set_alpha(0.55)
    ax.axhline(r["stage1_tau"], color="#333333", ls="--", lw=1.0,
               label=r"$\tau$ (normal 95%)")
    ax.set_ylabel("Stage-1 score $d$")
    legend_below(fig, ax, ncol=1, y=-0.02)
    ax.set_title("Dataset A: Stage-1 Mahalanobis score by class")
    fig.subplots_adjust(bottom=0.22, top=0.90, left=0.08, right=0.99)
    return save(fig, out_dir, "fig3_score_distribution")


if __name__ == "__main__":  # pragma: no cover
    print("fig3: score distribution")
