# -*- coding: utf-8 -*-
"""Table 4: feature-group ablation of the joint model."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .style import booktabs, by_name, write_tex_csv


def build(results: List[dict], out_dir: Path) -> List[Path]:
    """Generate ``table4_ablation.{tex,csv}``."""
    header = ["Dataset", "Stage 1 F1", "Cascade F1", "$\\Delta$F1"]
    rows, csv_rows = [], []
    for name in ["ding", "ds226", "lareunion"]:
        r = by_name(results).get(name)
        if not r or not r.get("has_label"):
            continue
        s1, c1 = r["stage1_f1"], r["cascade_f1"]
        rows.append([name, "%.3f" % s1, "%.3f" % c1, "%+.3f" % (c1 - s1)])
        csv_rows.append(dict(dataset=name, stage1_f1=s1, cascade_f1=c1, delta_f1=c1 - s1))
    tex = booktabs(header, rows,
                   caption="Layer ablation: change in F1 when the Stage-2 external-reference "
                           "residual rules are added to the Stage-1 probability-model decision.",
                   label="tab:ablation", colspec="lrrr")
    return write_tex_csv(out_dir / "table4_ablation", tex, csv_rows)


if __name__ == "__main__":  # pragma: no cover
    print("table4: ablation")
