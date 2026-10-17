# -*- coding: utf-8 -*-
"""Table 5: separability of the Stage-2 residual features (per-feature AUC for each dataset)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .style import booktabs, by_name, write_tex_csv

FEATS = ["r_min", "delta_r_min", "f_neg", "T_neg", "corr"]


def build(results: List[dict], out_dir: Path) -> List[Path]:
    """Generate ``table5_stage2_features.{tex,csv}``."""
    header = ["Dataset"] + [f.replace("_", "\\_") for f in FEATS]
    rows, csv_rows = [], []
    for name in ["ding", "ds226", "lareunion"]:
        r = by_name(results).get(name)
        if not r or not r.get("has_label"):
            continue
        au = r.get("stage2_feat_auc", {})
        vals = [au.get(f, np.nan) for f in FEATS]
        best = int(np.nanargmax([abs(v - 0.5) for v in vals])) if any(np.isfinite(vals)) else -1
        cells = []
        for i, v in enumerate(vals):
            s = "%.3f" % v if np.isfinite(v) else "--"
            cells.append("\\textbf{%s}" % s if i == best else s)
        rows.append([name] + cells)
        csv_rows.append(dict(dataset=name, **{f: vals[i] for i, f in enumerate(FEATS)}))
    tex = booktabs(header, rows,
                   caption="Per-feature AUC of the Stage-2 residual criteria "
                           "(bold = most discriminative, by $|\\mathrm{AUC}-0.5|$).",
                   label="tab:stage2_features", colspec="lrrrrr")
    return write_tex_csv(out_dir / "table5_stage2_features", tex, csv_rows)


if __name__ == "__main__":  # pragma: no cover
    print("table5: stage-2 features")
