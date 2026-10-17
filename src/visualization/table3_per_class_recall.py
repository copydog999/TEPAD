# -*- coding: utf-8 -*-
"""Table 3: single-layer feature and union-decision recall for each data label group of Ding."""
from __future__ import annotations

from pathlib import Path
from typing import List

from .style import booktabs, by_name, write_tex_csv

CLASS_ORDER = ["normal", "spike", "inverter_fault", "snowy", "cloudy", "low_value", "shading"]


def build(results: List[dict], out_dir: Path) -> List[Path]:
    """Generate ``table3_per_class_recall.{tex,csv}``."""
    r = by_name(results)["ding"]
    levels = r.get("per_class_levels", {})
    header = ["Class", "$n$", "Stage 1 only", "Stage 2 only",
              "Cascade ($s_1\\cup s_2$)"]
    rows, csv_rows = [], []
    for c in CLASS_ORDER:
        if c not in levels:
            continue
        v = levels[c]
        rows.append([c.replace("_", "\\_"), str(v.get("n", "")),
                     "%.3f" % v.get("stage1", 0.0), "%.3f" % v.get("stage2", 0.0),
                     "%.3f" % v.get("cascade", 0.0)])
        csv_rows.append(dict(cls=c, n=v.get("n"), stage1=v.get("stage1"),
                             stage2=v.get("stage2"), cascade=v.get("cascade")))
    tex = booktabs(header, rows,
                   caption="Per-label recall on the Ding dataset. The Stage-1 and Stage-2 "
                           "columns are per-layer diagnostics; only the cascade column is "
                           "the TEPAD decision.",
                   label="tab:per_class", colspec="lrrrr")
    return write_tex_csv(out_dir / "table3_per_class_recall", tex, csv_rows)


if __name__ == "__main__":  # pragma: no cover
    print("table3: per-class recall")
