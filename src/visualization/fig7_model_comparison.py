# -*- coding: utf-8 -*-
"""Figure 7: head-to-head model comparison on Dataset A (AUC and F1 bar charts)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt

from ..config import FIG_DOUBLE_COL
from .style import BEST_COLOR, COLORS, MUTED_COLOR, apply_style, load_baselines, save


def build(source, out_dir: Path, payload: Optional[dict] = None) -> List[Path]:
    """Generate ``fig7_model_comparison.{pdf,png}``.

    When ``payload`` is empty it is read from ``results/logs/baselines.json``; if that is missing, skip.
    """
    data = load_baselines(source, out_dir, payload)
    if not data:
        return []
    records = [r for r in data["models"]
               if "error" not in r]
    if not records:
        return []

    apply_style()
    names = [r["name"] for r in records]
    panels = [("AUC", [r.get("auc") for r in records]),
              ("F1", [r.get("f1") for r in records])]
    y = list(range(len(names)))

    fig, axes = plt.subplots(1, 2, figsize=(FIG_DOUBLE_COL, 0.34 * len(names) + 1.4),
                             sharey=True)
    for ax, (label, values) in zip(axes, panels):
        numeric = [float(v) if v is not None else float("nan") for v in values]
        best_index = int(max(range(len(numeric)),
                             key=lambda index: numeric[index]
                             if numeric[index] == numeric[index] else -float("inf")))
        colors = [BEST_COLOR if i == best_index else MUTED_COLOR
                  for i in range(len(values))]
        widths = [0.0 if v is None else max(float(v) - 0.5, 0.0)
                  for v in values]
        ax.barh(y, widths, left=0.5,
                height=0.55, color=colors)
        ax.set_xlim(0.5, 1.12)
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.3)
        for i, raw in zip(y, values):
            if raw is None:
                ax.annotate("n/a", (0.505, i), va="center", fontsize=6, color="#555555")
            else:
                ax.annotate("%.3f" % float(raw), (float(raw), i), xytext=(2, 0),
                            textcoords="offset points", va="center", fontsize=6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    fig.suptitle("Dataset A test split: model comparison (best value highlighted)",
                 fontsize=9)
    fig.subplots_adjust(left=0.24, right=0.99, bottom=0.07, top=0.91, wspace=0.22)
    return save(fig, out_dir, "fig7_model_comparison")


if __name__ == "__main__":  # pragma: no cover
    print("fig7: model comparison on Dataset A")
