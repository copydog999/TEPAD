"""Compact English figures for fault injections and stability (vector style)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd

from .. import config
from ..visualization.style import OVERLAP_LOGGER, check_text_overlap

BLUE = "#0072BD"
ORANGE = "#D95319"
YELLOW = "#EDB120"
PURPLE = "#7E2F8E"
GREEN = "#77AC30"
CYAN = "#4DBEEE"
RED = "#A2142F"
COLORS = [BLUE, ORANGE, YELLOW, PURPLE, GREEN, CYAN, RED]
BEST = "#77AC30"


def _bounded_rate_error(values, errors):
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    return np.vstack((np.minimum(errors, values), np.minimum(errors, 1.0 - values)))


def _style():
    # fontTools floods the log at INFO level while subsetting PDF fonts; quieten it here.
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.35,
        "axes.linewidth": 0.7,
        "axes.axisbelow": True,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.25,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 140,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _legend_below(fig, axes, ncol=3, bottom=0.20, extra=None):
    handles, labels = [], []
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    for ax in axes:
        current_handles, current_labels = ax.get_legend_handles_labels()
        for handle, label in zip(current_handles, current_labels):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
    if extra:
        for handle, label in extra:
            handles.append(handle)
            labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, 0.01), ncol=min(ncol, len(labels)),
                   frameon=False, fontsize=6.5, handlelength=1.5,
                   columnspacing=1.0)
        fig.subplots_adjust(bottom=bottom, top=0.91, left=0.10, right=0.98)


def _save(fig, path: Path, adjustments=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    overlaps = check_text_overlap(fig)
    if overlaps:
        if config.FIG_STRICT_OVERLAP:
            plt.close(fig)
            raise ValueError("Text overlap validation failed for {}: {}".format(
                path.name, "; ".join(overlaps)))
        OVERLAP_LOGGER.warning("Relaxed mode: saving %s despite %d overlap(s)",
                               path.name, len(overlaps))
    adjustment_note = "; ".join(adjustments) if adjustments else "none"
    OVERLAP_LOGGER.info("Layout adjustments for %s: %s", path.name, adjustment_note)
    if not path.name.startswith("sim_"):
        path = path.with_name("sim_" + path.name)
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight")
    fig.savefig(str(path.with_suffix(".png")), bbox_inches="tight")
    plt.close(fig)


def _box(ax, xy, width, height, text, color="#EAF2F8", edge=BLUE, fontsize=7):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.025,rounding_size=0.04",
        facecolor=color, edgecolor=edge, linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, wrap=True)
    return (x, y, width, height)


def _arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=8,
        linewidth=0.8, color="#444444",
    ))


def _flowchart(path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.05))
    titles = ["Strategy 1: Transient", "Strategy 2: Temporal", "Strategy 3: Mixed"]
    for ax, title in zip(axes, titles):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, pad=3, fontweight="bold")
        _box(ax, (0.05, 0.67), 0.25, 0.19, "1-min ticker\nG(t), T(t)", "#EAF2F8")
        _box(ax, (0.38, 0.67), 0.25, 0.19, "PVWatts\nPnormal(t)", "#EAF2F8")
        _box(ax, (0.71, 0.67), 0.24, 0.19, "Normal noise\nσ(t)", "#EAF2F8")
        _arrow(ax, (0.30, 0.765), (0.38, 0.765))
        _arrow(ax, (0.63, 0.765), (0.71, 0.765))
        if "Transient" in title:
            _box(ax, (0.26, 0.36), 0.48, 0.18, "Inject spike at t₁\nA = 2, 3.5, 5σ", "#FCE4D6", ORANGE)
        elif "Temporal" in title:
            _box(ax, (0.26, 0.36), 0.48, 0.18, "Shade [t₁,t₁+D)\ns = 0.7, 0.5, 0.3", "#FFF2CC", YELLOW)
        else:
            _box(ax, (0.26, 0.36), 0.48, 0.18, "Spike at t₁; shade at t₂>t₁\nD = 60 ticks", "#F4CCCC", RED)
        _arrow(ax, (0.83, 0.67), (0.70, 0.54))
        _box(ax, (0.20, 0.06), 0.60, 0.18, "Causal L=10 / L=60 detector\nlog t_inject, t_detect", "#E2F0D9", GREEN)
        _arrow(ax, (0.50, 0.36), (0.50, 0.24))
    fig.tight_layout(pad=0.4, w_pad=0.5)
    _save(fig, path)


def _system_architecture(path: Path):
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    items = [
        (0.01, "Time ticker\n1 tick = 1 min", "#EAF2F8", BLUE),
        (0.18, "Clear-sky\nG(t) × cloud", "#EAF2F8", BLUE),
        (0.35, "PVWatts\nG,T → P", "#EAF2F8", BLUE),
        (0.52, "Fault\ninjection", "#FCE4D6", ORANGE),
        (0.69, "L10 + L60\n+ residual-z", "#E2F0D9", GREEN),
        (0.86, "Detection delay\nand metrics", "#EAF2F8", BLUE),
    ]
    for x, text, color, edge in items:
        _box(ax, (x, 0.48), 0.135, 0.27, text, color, edge, 7)
    for i in range(len(items) - 1):
        _arrow(ax, (items[i][0] + 0.135, 0.615), (items[i + 1][0], 0.615))
    ax.text(0.5, 0.25,
            "Normal-only calibration: empirical 95th-percentile thresholds; "
            "EWMA parameter smoothing α = 0.95",
            ha="center", va="center", fontsize=8, color="#333333")
    ax.text(0.5, 0.06,
            "Every observation is consumed once in chronological order; no future samples are used.",
            ha="center", va="center", fontsize=7, style="italic")
    fig.tight_layout(pad=0.3)
    _save(fig, path)


def _example_injections(path: Path, theta: Dict[str, float], seed: int):
    from .model import (
        TrueParameters, inject_mixed, inject_shading, inject_spike, simulate_normal_day,
    )
    params = TrueParameters(**theta)
    day = simulate_normal_day(seed, 720, params)
    tick = 0.4 * 719
    t1 = int(round(tick))
    t2 = t1 + 30
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 5.0), sharex=True, sharey=True)
    g = day.irradiance / 1000.0
    normal = day.power_normal
    spike = inject_spike(day.power, t1, 3.5 * day.sigma[t1])
    shade = inject_shading(day.power, t1, 60, 0.5)
    mixed = inject_mixed(day.power, t1, 3.5 * day.sigma[t1], t2, 60, 0.5)
    axes[0].plot(day.ticks, normal, color="#777777", label="Normal")
    axes[0].plot(day.ticks, spike, color=ORANGE, label="Spike")
    axes[0].axvline(t1, color=ORANGE, linestyle="--", linewidth=0.8,
                     label="Injection onset")
    axes[0].set_title("Strategy 1: 3.5σ one-tick spike")
    axes[1].plot(day.ticks, normal, color="#777777", label="Normal")
    axes[1].plot(day.ticks, shade, color=BLUE, label="Shading")
    axes[1].axvspan(t1, t1 + 60, color=BLUE, alpha=0.13,
                    label="Shading interval")
    axes[1].set_title("Strategy 2: s=0.5, D=60 ticks")
    axes[2].plot(day.ticks, normal, color="#777777", label="Normal")
    axes[2].plot(day.ticks, mixed, color=PURPLE, label="Mixed")
    axes[2].axvline(t1, color=ORANGE, linestyle="--", linewidth=0.8,
                    label="Spike onset")
    axes[2].axvspan(t2, t2 + 60, color=PURPLE, alpha=0.12,
                    label="Shading interval")
    axes[2].set_title("Strategy 3: spike then shading")
    axes[2].set_xlabel("Time (ticks; 1 tick = 1 min)")
    for index, ax in enumerate(axes):
        ax.set_ylabel("Normalized power" if index == 0 else "", labelpad=12)
        ax.tick_params(axis="y", labelleft=index == 0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.grid(True)
        ax.set_xlim(0, 719)
    _legend_below(fig, axes, ncol=3, bottom=0.19)
    fig.tight_layout(rect=(0, 0.18, 1, 1), h_pad=1.0)
    _save(
        fig, path,
        adjustments=(
            "shared the power scale and removed repeated y-axis labels/tick labels",
            "increased vertical subplot spacing and figure height",
        ),
    )


def _injection_figures(path: Path, injections: Dict[str, object]):
    spike = injections["spike"]
    shade = injections["shading"]
    mixed = injections["mixed"]
    levels = ["low", "medium", "high"]
    labels = ["Low", "Medium", "High"]
    positions = np.arange(3)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75))
    groups = (
        (spike, BLUE, -0.12, "Spike"),
        (mixed, ORANGE, 0.12, "Mixed: spike"),
    )
    spike_rates = [row["detection_rate_mean"] for row in spike]
    mixed_spike_rates = [row["spike"]["detection_rate_mean"] for row in mixed]
    mixed_shade_rates = [row["shading"]["detection_rate_mean"] for row in mixed]
    best_spike_index = int(np.argmax(spike_rates))
    best_mixed_index = int(np.argmax(mixed_spike_rates))
    best_shade_index = int(np.argmax(mixed_shade_rates))
    for group, color, shift, label in groups:
        rates = [row["detection_rate_mean"] if "detection_rate_mean" in row
                 else row["spike"]["detection_rate_mean"] for row in group]
        errors = [row["detection_rate_sd"] if "detection_rate_sd" in row
                  else row["spike"]["detection_rate_sd"] for row in group]
        best_index = best_spike_index if label == "Spike" else best_mixed_index
        axes[0].bar(positions + shift, rates, width=0.22,
                    yerr=_bounded_rate_error(rates, errors),
                    error_kw={"capsize": 2},
                    color=[BEST if i == best_index else color for i in range(len(rates))],
                    label=label)
    mixed_shade_errors = [row["shading"]["detection_rate_sd"] for row in mixed]
    axes[0].bar(positions + 0.37, mixed_shade_rates,
                width=0.22, yerr=_bounded_rate_error(mixed_shade_rates, mixed_shade_errors),
                error_kw={"capsize": 2},
                color=[BEST if i == best_shade_index else YELLOW for i in range(3)],
                label="Mixed: shading")
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Event detection rate")
    axes[0].set_title("Detection by severity (mean ± SD)")
    axes[0].grid(axis="y")

    shade_order = [
        next(row for row in shade if row["level"] == level and row["duration"] == duration)
        for level in levels for duration in ("short", "long")
    ]
    x = np.arange(len(shade_order))
    delay_medians = [r["delay_median_ticks"] for r in shade_order]
    delay_best = (int(np.nanargmin([np.nan if value is None else value
                                    for value in delay_medians]))
                  if any(value is not None for value in delay_medians) else -1)
    rate_ax = axes[1]
    delay_ax = rate_ax.twinx()
    rate_values = [r["detection_rate_mean"] for r in shade_order]
    rate_errors = [r["detection_rate_sd"] for r in shade_order]
    rate_ax.bar(x - 0.16, rate_values,
                width=0.32, yerr=_bounded_rate_error(rate_values, rate_errors),
                capsize=2,
                color=[BEST if i == int(np.argmax(rate_values)) else CYAN
                       for i in range(len(rate_values))],
                label="Detection rate")
    delay_ax.scatter(
        x + 0.16, [r["delay_median_ticks"] if r["delay_median_ticks"] is not None else np.nan
                   for r in shade_order],
        color=[BEST if i == delay_best else RED for i in range(len(x))],
        marker="D", s=20, label="Median delay",
    )
    rate_ax.set_xticks(x, ["{}-{}".format(r["level"][:1].upper(), r["duration"][0].upper())
                           for r in shade_order])
    rate_ax.set_ylabel("Detection rate")
    rate_ax.set_ylim(0, 1.08)
    rate_ax.set_title("Shading severity × duration")
    rate_ax.grid(axis="y")
    delay_ax.set_ylabel("Median delay (ticks)")
    finite_delays = [value for value in delay_medians if value is not None]
    delay_upper = (max(finite_delays) * 1.25
                   if finite_delays and max(finite_delays) > 0 else 1.0)
    delay_ax.set_ylim(-0.04 * delay_upper, delay_upper)
    delay_ax.set_yticks(np.linspace(0, delay_upper, 6))
    _legend_below(fig, [axes[0], rate_ax, delay_ax], ncol=4, bottom=0.23,
                  extra=[(Patch(facecolor=BEST), "Best result")])
    fig.tight_layout(rect=(0, 0.22, 1, 1), pad=0.5, w_pad=1.0)
    _save(fig, path)


def _delay_cdf(path: Path, trials_path: Path):
    frame = pd.read_csv(trials_path)
    groups = []
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    curves = []
    for strategy, score_column in (
        ("spike", "delay_ticks"),
        ("shading", "delay_ticks"),
        ("mixed spike", "spike_delay_ticks"),
        ("mixed shading", "shading_delay_ticks"),
    ):
        selected = frame[frame["strategy"] == strategy.split()[0]]
        if strategy.startswith("mixed "):
            selected = selected[selected["strategy"] == "mixed"]
        if not len(selected) or score_column not in selected:
            continue
        for level, group in selected.groupby("level", dropna=False):
            values = pd.to_numeric(
                group[score_column], errors="coerce"
            ).dropna().to_numpy()
            if not len(values):
                continue
            values.sort()
            y = np.arange(1, len(values) + 1) / len(values)
            label = "{}-{}".format(strategy, level)
            curves.append((values, y, label, strategy, level))
            groups.append((strategy, level))
    fastest = min(curves, key=lambda item: float(np.median(item[0])))[2] if curves else None
    for values, y, label, strategy, level in curves:
        color = BEST if label == fastest else COLORS[len(ax.lines) % len(COLORS)]
        ax.step(values, y, where="post", label=label, color=color)
    ax.set_xlabel("Detection delay (ticks)")
    ax.set_ylabel("Empirical CDF (detected trials)")
    ax.set_title("Detection-delay distributions")
    ax.set_ylim(0, 1.02)
    ax.grid(True)
    _legend_below(fig, ax, ncol=4, bottom=0.24,
                  extra=[(Patch(color=BEST), "Lowest median delay")])
    fig.tight_layout(rect=(0, 0.22, 1, 1), pad=0.4)
    _save(fig, path)


def _window_plot(path: Path, frame: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for i, parameter in enumerate(("alpha", "beta", "eta", "sigma0", "sigma_min")):
        sub = frame[frame["parameter"] == parameter].sort_values("condition")
        ax.plot(sub["condition"], sub["mse"], marker="o", markersize=3,
                color=COLORS[i], label=parameter)
        best = sub.loc[sub["mse"].idxmin()]
        ax.scatter([best["condition"]], [best["mse"]], color=BEST, marker="*",
                   s=38, zorder=4)
    ax.set_xlabel("Window length L (samples)")
    ax.set_ylabel("Estimator MSE")
    ax.set_yscale("log")
    ax.set_title("Parameter-estimation MSE vs. window length")
    ax.grid(True, which="both")
    _legend_below(fig, ax, ncol=5, bottom=0.21,
                  extra=[(Line2D([], [], color=BEST, marker="*", linestyle="None"),
                          "Lowest MSE")])
    fig.tight_layout(rect=(0, 0.19, 1, 1), pad=0.4)
    _save(fig, path)


def _baseline_stability_plot(path: Path, frame: pd.DataFrame):
    parameters = ("alpha", "beta", "eta", "sigma0", "sigma_min")
    indexed = frame.set_index("parameter").loc[list(parameters)]
    x = np.arange(len(parameters))
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.55))
    bias_colors = [BEST if abs(value) == min(abs(indexed["bias"])) else BLUE
                   for value in indexed["bias"]]
    axes[0].bar(x, indexed["bias"], color=bias_colors, width=0.62)
    axes[0].axhline(0, color="#555555", linewidth=0.7)
    axes[0].set_ylabel("Bias")
    axes[0].set_title("Estimator bias (L=30)")
    axes[0].set_xticks(x, parameters, rotation=20)
    axes[0].grid(axis="y")
    variance_colors = [BEST if value == min(indexed["variance"]) else ORANGE
                       for value in indexed["variance"]]
    axes[1].bar(x, indexed["variance"], color=variance_colors, width=0.62)
    axes[1].set_ylabel("Error variance")
    axes[1].set_title("Estimator variance (L=30)")
    axes[1].set_xticks(x, parameters, rotation=20)
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", which="both")
    _legend_below(fig, axes, ncol=1, bottom=0.19,
                  extra=[(Patch(facecolor=BEST), "Lowest error")])
    fig.tight_layout(rect=(0, 0.17, 1, 1), pad=0.45, w_pad=0.8)
    _save(fig, path)


def _noise_plot(path: Path, frame: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for i, parameter in enumerate(("eta", "sigma0", "sigma_min")):
        sub = frame[frame["parameter"] == parameter].sort_values("condition")
        ax.plot(sub["condition"], sub["mse"], marker="o", markersize=3,
                color=COLORS[i], label=parameter)
        best = sub.loc[sub["mse"].idxmin()]
        ax.scatter([best["condition"]], [best["mse"]], color=BEST, marker="*",
                   s=38, zorder=4)
    ax.set_xlabel("Noise standard deviation σ")
    ax.set_ylabel("Estimator MSE")
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.set_yscale("log")
    ax.set_title("Noise-level sensitivity (N=100)")
    ax.grid(True, which="both")
    _legend_below(fig, ax, ncol=4, bottom=0.22,
                  extra=[(Line2D([], [], color=BEST, marker="*", linestyle="None"),
                          "Lowest MSE")])
    fig.tight_layout(rect=(0, 0.20, 1, 1), pad=0.4)
    _save(
        fig, path,
        adjustments=("used compact significant-value formatting on the noise axis",),
    )


def _contamination_plot(path: Path, frame: pd.DataFrame):
    parameters = ("eta", "sigma0", "alpha")
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.8))
    for i, (ax, parameter) in enumerate(zip(axes, parameters)):
        sub = frame[frame["parameter"] == parameter].sort_values("condition")
        ax.plot(sub["condition"], sub["bias"], marker="o", markersize=3,
                color=COLORS[i])
        best = sub.loc[sub["bias"].abs().idxmin()]
        ax.scatter([best["condition"]], [best["bias"]], color=BEST, marker="*",
                   s=38, zorder=4)
        ax.axhline(0, color="#555555", linestyle="--", linewidth=0.7)
        ax.set_title(parameter)
        ax.set_xlabel("Injected spike (σ)")
        ax.set_ylabel("Bias")
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True)
    fig.suptitle("Single-point anomaly contamination (N=100)", y=0.99)
    fig.legend([Line2D([], [], color=BEST, marker="*", linestyle="None")],
               ["Lowest absolute bias"], loc="lower center",
               bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=6.5)
    fig.subplots_adjust(bottom=0.23, top=0.80, left=0.10, right=0.99, wspace=0.65)
    _save(
        fig, path,
        adjustments=(
            "reduced tick-label size and widened panel spacing",
            "added clearance between the suptitle, axes, and shared legend",
        ),
    )


def _convergence_plot(path: Path, frame: pd.DataFrame):
    frame = frame.sort_values("condition")
    fig, ax1 = plt.subplots(figsize=(4.4, 2.8))
    ax1.plot(frame["condition"], frame["bias"], marker="o", color=BLUE,
             label="Bias(η)")
    ax1.axhline(0, color=BLUE, linestyle="--", linewidth=0.7)
    ax1.set_xscale("log")
    ax1.set_xlabel("Sample count N")
    ax1.set_ylabel("Bias(η)", color=BLUE)
    ax2 = ax1.twinx()
    ax2.plot(frame["condition"], frame["variance"], marker="s", color=ORANGE,
             label="Var(η)")
    best = frame.loc[frame["variance"].idxmin()]
    ax2.scatter([best["condition"]], [best["variance"]], color=BEST, marker="*",
                s=42, zorder=4, label="Lowest variance")
    ax2.set_yscale("log")
    ax2.set_ylabel("Var(η)", color=ORANGE)
    ax1.grid(True, which="both")
    ax1.set_title("Estimator convergence (100 repetitions)")
    _legend_below(fig, [ax1, ax2], ncol=3, bottom=0.23)
    fig.tight_layout(rect=(0, 0.21, 1, 1), pad=0.4)
    _save(fig, path)


def _definition_plot(path: Path, definitions: Dict[str, object]):
    """E3: detection rate of the two shading definitions (including the endogenous-channel ablation), MTD and noise sensitivity."""
    full = {r["definition"]: r for r in definitions["by_definition"]}
    endogenous = {r["definition"]: r for r
                  in definitions.get("ablation_endogenous", {})
                  .get("by_definition", [])}
    keys = [k for k in ("pv_only", "pv_and_sensor") if k in full]
    labels = ["Def. #1\nPV only", "Def. #2\nPV + sensor"][:len(keys)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))

    x = np.arange(len(keys))
    width = 0.38
    full_rates = [full[k]["detection_rate"] for k in keys]
    endo_rates = [endogenous.get(k, {}).get("detection_rate", np.nan) for k in keys]
    ceiling = max([v for v in full_rates + endo_rates
                   if v == v] or [0.0])
    axes[0].bar(x - width / 2, full_rates, width,
                color=[BEST if v == ceiling else BLUE for v in full_rates],
                label="Full detector")
    axes[0].bar(x + width / 2, [0 if v != v else v for v in endo_rates], width,
                color=[BEST if v == ceiling else ORANGE for v in endo_rates],
                label="Endogenous only")
    for xi, value in zip(x - width / 2, full_rates):
        axes[0].text(xi, value + 0.02, "%.3f" % value, ha="center", fontsize=6)
    for xi, value in zip(x + width / 2, endo_rates):
        if value == value:
            axes[0].text(xi, value + 0.02, "%.3f" % value, ha="center", fontsize=6)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Detection rate")
    axes[0].set_title("Shading definition")
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False, fontsize=6, loc="lower left")

    mtds = [full[k]["mtd_ticks"] if full[k]["mtd_ticks"] is not None else np.nan
            for k in keys]
    finite = [v for v in mtds if v == v]
    best_mtd = int(np.nanargmin(mtds)) if finite else -1
    axes[1].bar(x, [0 if v != v else v for v in mtds], width=0.55,
                color=[BEST if i == best_mtd else ORANGE for i in range(len(keys))])
    for i, value in enumerate(mtds):
        axes[1].text(i, (0 if value != value else value) + 0.4,
                     "n/a" if value != value else "%.2f" % value,
                     ha="center", fontsize=6.5)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("MTD (ticks)")
    axes[1].set_ylim(0, max(finite or [0.0]) + 1.0)
    axes[1].set_title("Mean detection time")
    axes[1].grid(axis="y")

    noise_rows = definitions["noise_sweep"]["rows"]
    ceiling_noise = max([r["detection_rate"] for r in noise_rows] or [0.0])
    for index, (definition, label) in enumerate((("pv_only", "Def. #1"),
                                                 ("pv_and_sensor", "Def. #2"))):
        subset = sorted([r for r in noise_rows if r["definition"] == definition],
                        key=lambda r: r["noise_sigma"])
        values = [r["detection_rate"] for r in subset]
        color = BEST if max(values or [0.0]) == ceiling_noise else COLORS[index]
        axes[2].plot([r["noise_sigma"] for r in subset], values, marker="o",
                     markersize=3.5, color=color, label=label)
    axes[2].set_xlabel(r"Measurement noise $\sigma$")
    axes[2].set_ylabel("Detection rate")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Noise sensitivity")
    axes[2].grid(True)
    axes[2].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1), pad=0.4, w_pad=0.9)
    _save(
        fig, path,
        adjustments=("added headroom above mean-detection-time value labels",),
    )


def _timeliness_plot(path: Path, timeliness: Dict[str, object]):
    """E5: DTM-tau curves versus MTD."""
    summary = timeliness["summary"]
    thresholds = timeliness["thresholds"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    dtm_curves = {}
    for entry in summary:
        dtm_curves[entry["anomaly_type"]] = [
            entry["dtm_{}".format(tau)] for tau in thresholds
        ]
    best_rate = max(sum(curve) for curve in dtm_curves.values())
    best_curve = max(dtm_curves, key=lambda name: sum(dtm_curves[name]))
    for index, (name, curve) in enumerate(dtm_curves.items()):
        color = BEST if name == best_curve and sum(curve) == best_rate \
            else COLORS[index % len(COLORS)]
        axes[0].plot(thresholds, curve, marker="o", markersize=3.5, color=color,
                     label=name)
    axes[0].set_xlabel(r"Delay threshold $\tau$ (ticks)")
    axes[0].set_ylabel(r"DTM($\tau$)")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xscale("log")
    axes[0].set_title("Detection within tolerance")
    axes[0].grid(True, which="both")

    names = [entry["anomaly_type"] for entry in summary]
    mtds = [entry["mtd_ticks"] if entry["mtd_ticks"] is not None else 0.0
            for entry in summary]
    best_mtd_index = int(np.argmin([v if v > 0 else np.inf for v in mtds])) \
        if any(v > 0 for v in mtds) else -1
    axes[1].bar(np.arange(len(names)), mtds, width=0.6,
                color=[BEST if i == best_mtd_index else PURPLE for i in range(len(names))])
    for i, value in enumerate(mtds):
        axes[1].text(i, value + 0.4, "%.1f" % value, ha="center", fontsize=6.5)
    axes[1].set_xticks(np.arange(len(names)), names, rotation=15)
    axes[1].set_ylabel("MTD (ticks)")
    axes[1].set_title("Mean time to detect")
    axes[1].grid(axis="y")
    _legend_below(fig, axes, ncol=5, bottom=0.25)
    fig.tight_layout(rect=(0, 0.23, 1, 1), pad=0.4, w_pad=0.9)
    _save(fig, path)


def make_timeliness_figures(definitions: Dict[str, object],
                            timeliness: Dict[str, object],
                            figures_dir: Path):
    """Generate the E3 / E5 figures (PDF + PNG)."""
    _style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    _definition_plot(figures_dir / "sim_shading_definitions", definitions)
    _timeliness_plot(figures_dir / "sim_timeliness", timeliness)


def make_simulation_figures(result: Dict[str, object],
                            stability: Dict[str, pd.DataFrame],
                            figures_dir: Path):
    _style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    _system_architecture(figures_dir / "sim_system_architecture")
    _flowchart(figures_dir / "sim_injection_flowcharts")
    _example_injections(figures_dir / "sim_injection_examples",
                        result["theta_true"], seed=781)
    _injection_figures(figures_dir / "sim_detection_and_delay",
                       result["injections"])
    _delay_cdf(figures_dir / "sim_delay_cdf",
               figures_dir.parent / "tables" / "simulation_trials.csv")
    _baseline_stability_plot(
        figures_dir / "sim_stability_baseline_bias_variance",
        stability["bias_variance"],
    )
    _window_plot(figures_dir / "sim_stability_window_mse", stability["window"])
    _noise_plot(figures_dir / "sim_stability_noise_mse", stability["noise"])
    _contamination_plot(figures_dir / "sim_stability_contamination",
                        stability["contamination"])
    _convergence_plot(figures_dir / "sim_stability_convergence",
                      stability["convergence"])
