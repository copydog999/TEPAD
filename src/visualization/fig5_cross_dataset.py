# -*- coding: utf-8 -*-
"""Figure 5: cross-dataset comparison (Stage-1 AUC / cascade F1 / anomaly ratio)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .style import BEST_COLOR, COLORS, by_name, dataset_label, new_fig, save

ORDER = ["ding", "ds226", "lareunion"]


def build(results: List[dict], out_dir: Path) -> List[Path]:
    """Generate ``fig5_cross_dataset.{pdf,png}``."""
    d = by_name(results)
    names = [n for n in ORDER if n in d]
    auc = [d[n]["stage1_auc"] if np.isfinite(d[n]["stage1_auc"]) else np.nan for n in names]
    f1 = [d[n].get("cascade_f1", np.nan) for n in names]
    ratio = [d[n]["anomaly_ratio"] for n in names]

    x = np.arange(len(names))
    w = 0.38
    fig, ax = new_fig(double=True, height=2.8)
    best_auc = int(np.nanargmax(auc)) if np.isfinite(auc).any() else -1
    best_f1 = int(np.nanargmax(f1)) if np.isfinite(f1).any() else -1
    ax.bar(x - w / 2, np.asarray(auc) - 0.5, w, bottom=0.5, label="Stage-1 AUC",
           color=[BEST_COLOR if i == best_auc else COLORS[0] for i in range(len(names))])
    ax.bar(x + w / 2, np.asarray(f1) - 0.5, w, bottom=0.5,
           label="Cascade F1",
           color=[BEST_COLOR if i == best_f1 else COLORS[1] for i in range(len(names))])
    for xi, v in zip(x - w / 2, auc):
        if np.isfinite(v):
            ax.text(xi, v + 0.012, "%.2f" % v, ha="center", fontsize=6)
    for xi, v in zip(x + w / 2, f1):
        if np.isfinite(v):
            ax.text(xi, v + 0.012, "%.2f" % v, ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([dataset_label(name) for name in names])
    ax.set_ylim(0.5, 1.08)
    ax.set_ylabel("Score")
    ax2 = ax.twinx()
    ax2.plot(x, ratio, "o--", color=COLORS[3], ms=3, lw=1.0, label="Anomaly ratio")
    ax2.set_ylabel("Anomaly ratio")
    ax2.set_ylim(0, max(0.25, max(ratio) * 1.3))
    ax2.grid(False)
    handles = [
        Patch(facecolor=COLORS[0], label="Stage-1 AUC"),
        Patch(facecolor=COLORS[1], label="Cascade F1"),
        Line2D([], [], color=COLORS[3], marker="o", linestyle="--",
               label="Anomaly ratio"),
        Patch(facecolor=BEST_COLOR, label="Best metric"),
    ]
    labels_legend = [handle.get_label() for handle in handles]
    fig.legend(handles, labels_legend, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=6)
    ax.set_title("Cross-dataset comparison")
    fig.subplots_adjust(bottom=0.23, top=0.90, left=0.08, right=0.90)
    return save(fig, out_dir, "fig5_cross_dataset")


if __name__ == "__main__":  # pragma: no cover
    print("fig5: cross_dataset")
