# -*- coding: utf-8 -*-
"""Figure 6: separability of the Stage-2 residual features (per-feature AUC for each dataset)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from matplotlib.patches import Patch

from .style import BEST_COLOR, COLORS, by_name, dataset_label, new_fig, save

FEATS = ["r_min", "delta_r_min", "f_neg", "T_neg", "corr"]


def build(results: List[dict], out_dir: Path) -> List[Path]:
    """Generate ``fig6_stage2_features.{pdf,png}``."""
    names = [n for n in ["ding", "ds226", "lareunion"] if n in by_name(results)]
    if not names:
        return []
    x = np.arange(len(FEATS))
    w = 0.8 / max(len(names), 1)
    fig, ax = new_fig(double=True, height=2.8)
    values_by_dataset = {}
    for i, n in enumerate(names):
        au = by_name(results)[n].get("stage2_feat_auc", {})
        values_by_dataset[n] = [au.get(f, np.nan) for f in FEATS]
    for i, n in enumerate(names):
        vals = values_by_dataset[n]
        bars = ax.bar(
            x + i * w - 0.4 + w / 2,
            [v - 0.5 if np.isfinite(v) else 0.0 for v in vals],
            w, bottom=0.5, label=dataset_label(n), color=COLORS[i],
        )
        for j, bar in enumerate(bars):
            candidates = [(abs(values_by_dataset[name][j] - 0.5), index)
                          for index, name in enumerate(names)
                          if np.isfinite(values_by_dataset[name][j])]
            if candidates and max(candidates)[1] == i:
                bar.set_facecolor(BEST_COLOR)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax.text(len(FEATS) - 0.42, 0.515, "random", fontsize=6, color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in FEATS])
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.08)
    handles = [Patch(facecolor=COLORS[i], label=dataset_label(name))
               for i, name in enumerate(names)]
    handles.append(Patch(facecolor=BEST_COLOR, label="Most separable from chance"))
    labels = [handle.get_label() for handle in handles]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, frameon=False, fontsize=6)
    ax.set_title("Stage-2 residual feature separability")
    fig.subplots_adjust(bottom=0.25, top=0.90, left=0.08, right=0.99)
    return save(fig, out_dir, "fig6_stage2_features")


if __name__ == "__main__":  # pragma: no cover
    print("fig6: stage-2 features")
