# -*- coding: utf-8 -*-
"""Shared plotting/table style: Times New Roman, compact layout, dual-format output."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text

from ..config import (FIG_DPI, FIG_DOUBLE_COL, FIG_SINGLE_COL, FIG_STRICT_OVERLAP,
                      FONT_FAMILY, FONT_SIZE, TITLE_SIZE)

#: Colour-blind-friendly palette (taken from matplotlib tab10)
COLORS: List[str] = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
BEST_COLOR = "#77AC30"
MUTED_COLOR = "#AEB8C2"
DATASET_LABELS = {
    "ding": "Dataset A",
    "ds226": "Dataset B",
    "lareunion": "Dataset C",
}
OVERLAP_LOGGER = logging.getLogger("src.visualization.text_overlap")


def apply_style() -> None:
    """Set the global plotting style (Times New Roman + compact typesetting)."""
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [FONT_FAMILY],
        "mathtext.fontset": "stix",
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.linewidth": 0.7,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.45,
        "axes.unicode_minus": False,
        "figure.constrained_layout.use": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def new_fig(double: bool = False, height: float = None):
    """Create a new figure; ``double=True`` uses the double-column width (7 in), otherwise single-column (3.5 in)."""
    width = FIG_DOUBLE_COL if double else FIG_SINGLE_COL
    return plt.subplots(figsize=(width, height if height else width * 0.62))


def _text_polygon(artist: Text, renderer):
    bounds = artist.get_window_extent(renderer)
    angle = math.radians(artist.get_rotation())
    cosine, sine = abs(math.cos(angle)), abs(math.sin(angle))
    determinant = cosine * cosine - sine * sine
    if abs(determinant) < 1e-6:
        width, height = bounds.width, bounds.height
    else:
        width = (bounds.width * cosine - bounds.height * sine) / determinant
        height = (bounds.height * cosine - bounds.width * sine) / determinant
    center_x = (bounds.x0 + bounds.x1) / 2.0
    center_y = (bounds.y0 + bounds.y1) / 2.0
    corners = []
    for x, y in ((-width / 2, -height / 2), (-width / 2, height / 2),
                 (width / 2, height / 2), (width / 2, -height / 2)):
        corners.append((
            center_x + x * math.cos(angle) - y * math.sin(angle),
            center_y + x * math.sin(angle) + y * math.cos(angle),
        ))
    return bounds, corners


def _text_polygons_overlap(first, second, tolerance: float = 0.75) -> bool:
    for polygon in (first, second):
        for index, start in enumerate(polygon):
            end = polygon[(index + 1) % len(polygon)]
            axis_x, axis_y = -(end[1] - start[1]), end[0] - start[0]
            first_projection = [x * axis_x + y * axis_y for x, y in first]
            second_projection = [x * axis_x + y * axis_y for x, y in second]
            if (max(first_projection) <= min(second_projection) + tolerance
                    or max(second_projection) <= min(first_projection) + tolerance):
                return False
    return True


def dataset_label(name: str) -> str:
    """Return the anonymized paper label for a supported dataset."""
    normalized = name.strip().lower().replace(" ", "").replace("-", "")
    if normalized.startswith("dataset"):
        return name
    try:
        return DATASET_LABELS[normalized]
    except KeyError as exc:
        raise ValueError("No anonymized label configured for dataset {!r}".format(name)) from exc


def check_text_overlap(fig) -> List[str]:
    """Find intersecting visible text bounding boxes in a rendered figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = []
    hidden_tick_ids = {
        id(label)
        for axes in fig.axes if not axes.axison
        for label in axes.get_xticklabels() + axes.get_yticklabels()
    }
    for artist in fig.findobj(match=Text):
        if (id(artist) in hidden_tick_ids or not artist.get_visible()
                or not artist.get_text().strip()):
            continue
        bounds, polygon = _text_polygon(artist, renderer)
        if bounds.width <= 1.0 or bounds.height <= 1.0:
            continue
        labels.append((artist.get_text().replace("\n", " / "), bounds, polygon))

    overlaps = []
    for index, (first_label, first_bounds, first_polygon) in enumerate(labels):
        for second_label, second_bounds, second_polygon in labels[index + 1:]:
            if _text_polygons_overlap(first_polygon, second_polygon):
                first_position = tuple(round(value, 1) for value in (
                    first_bounds.x0, first_bounds.y0, first_bounds.x1, first_bounds.y1))
                second_position = tuple(round(value, 1) for value in (
                    second_bounds.x0, second_bounds.y0, second_bounds.x1, second_bounds.y1))
                overlaps.append("{} {} <> {} {}".format(
                    first_label, first_position, second_label, second_position))

    if overlaps:
        OVERLAP_LOGGER.error("FAIL: %d text overlap(s): %s",
                             len(overlaps), "; ".join(overlaps))
    else:
        OVERLAP_LOGGER.info("PASS: %d text elements; no overlaps; adjustments: none",
                            len(labels))
    return overlaps


def save(fig, out_dir: Path, name: str,
         adjustments: Sequence[str] = ()) -> List[Path]:
    """Validate and save a data figure as anonymized PDF and PNG files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlaps = check_text_overlap(fig)
    if overlaps:
        if FIG_STRICT_OVERLAP:
            plt.close(fig)
            raise ValueError("Text overlap validation failed for {}: {}".format(
                name, "; ".join(overlaps)))
        OVERLAP_LOGGER.warning("Relaxed mode: saving %s despite %d overlap(s)",
                               name, len(overlaps))
    adjustment_note = "; ".join(adjustments) if adjustments else "none"
    OVERLAP_LOGGER.info("Layout adjustments for %s: %s", name, adjustment_note)
    output_name = name if name.startswith(("dat_", "sim_")) else "dat_" + name
    pdf = out_dir / (output_name + ".pdf")
    png = out_dir / (output_name + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=FIG_DPI)
    plt.close(fig)
    return [pdf, png]


def legend_below(fig, axes, ncol: int = 3, fontsize: float = None,
                 y: float = 0.0):
    """Place one shared legend below the axes, never over plotted data."""
    axes = list(axes) if isinstance(axes, (list, tuple)) else [axes]
    handles, labels = [], []
    for ax in axes:
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
    if not handles:
        return None
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y),
               ncol=min(ncol, len(labels)), frameon=False,
               fontsize=fontsize or FONT_SIZE - 1, handlelength=1.5,
               columnspacing=1.1)
    return fig


def booktabs(header: Sequence[str], rows: Iterable[Sequence[str]], caption: str,
             label: str, colspec: str, note: str = "") -> str:
    """Build a booktabs-style LaTeX table string.

    Parameters
    ----------
    header : Sequence[str]
        Header cells.
    rows : Iterable[Sequence[str]]
        Data rows (already formatted strings).
    caption, label : str
        Caption and ``\\label``.
    colspec : str
        ``tabular`` column spec, e.g. ``"llrr"``.
    note : str
        Table note placed below the table.
    """
    lines = ["% auto-generated, do not edit by hand", "\\begin{table}[t]", "  \\centering",
             "  \\caption{%s}" % caption, "  \\label{%s}" % label,
             "  \\begin{tabular}{%s}" % colspec, "    \\toprule",
             "    " + " & ".join(header) + " \\\\", "    \\midrule"]
    for r in rows:
        lines.append("    " + " & ".join(str(x) for x in r) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}"]
    if note:
        lines.append("  \\tablenotes{\\footnotesize %s}" % note)
    lines += ["\\end{table}", ""]
    return "\n".join(lines)


def write_tex_csv(path_stem: Path, tex: str, rows_csv) -> List[Path]:
    """Write both the ``.tex`` and the ``.csv`` version."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    tex_path = path_stem.with_suffix(".tex")
    csv_path = path_stem.with_suffix(".csv")
    tex_path.write_text(tex, encoding="utf-8")
    try:
        import pandas as pd
        pd.DataFrame(rows_csv).to_csv(csv_path, index=False)
    except Exception:  # pragma: no cover
        pass
    return [tex_path, csv_path]


def by_name(results):
    """Turn the result list into ``{dataset: result}``."""
    return {r["dataset"]: r for r in results}


def load_baselines(source, out_dir, payload: Optional[dict] = None) -> Optional[dict]:
    """Load the comparison-experiment results (shared by Table 6 / Figure 7).

    An explicit ``payload`` takes precedence; otherwise the results are read from
    ``<results>/logs/baselines.json`` (if ``source`` itself is a dict containing
    ``models``, it is used directly as well).
    Returns ``None`` when the comparison experiments have not been run, in which case
    the caller should skip.
    """
    if payload is not None:
        return payload
    if isinstance(source, dict) and "models" in source:
        return source
    path = Path(out_dir).parent / "logs" / "baselines.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("models") else None


if __name__ == "__main__":  # pragma: no cover - minimal example
    apply_style()
    print(booktabs(["A", "B"], [[1, 2]], "cap", "lab", "rr")[:120])
