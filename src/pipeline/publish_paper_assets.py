"""Publish one curated set of manuscript figures and tables."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402

#: One-to-one correspondence with the ``\includegraphics`` file names in ``paper.tex``.
#: ``tepadarch`` and ``2stage`` are the hand-drawn method diagrams of the paper (provided by
#: ``research_paper/paper/solar_energy/figures/``); this pipeline only **keeps** them and
#: does not redraw them, so they must also be listed here, otherwise they would be archived
#: as legacy artifacts.
FIGURES = (
    "tepadarch",
    "2stage",
    "sim_injection_examples",
    "sim_interpretability",
    "sim_results",
    "sim_stability",
    "dat_model_pipeline",
    "dat_parameter_distribution",
)
TABLES = (
    "sim_interpretability",
    "sim_detection_classification_timeliness",
    "sim_stability",
    "dat_main",
    "dat_cross_dataset",
    "dat_baseline",
    "dat_parameter",
)
PARAMETERS = ("alpha", "beta", "eta", "sigma0", "r_min")
PARAMETER_LABELS = (r"$\alpha$", r"$\beta$", r"$\eta$", r"$\sigma_0$", r"$r_{\min}$")
CLASS_LABELS = {
    "hardware": "Hardware anomaly (power-conversion event)",
    "weather": "Weather anomaly (irradiance-input event)",
    "shading": "Shading anomaly (module-reception event)",
}
DATASET_LABELS = {"ding": "Dataset A", "ds226": "Dataset B", "lareunion": "Dataset C"}
STABILITY_FILES = (
    ("Bias and variance", "simulation_stability_bias_variance.csv"),
    ("Window length", "simulation_stability_window.csv"),
    ("Measurement noise", "simulation_stability_noise.csv"),
    ("Spike contamination", "simulation_stability_contamination.csv"),
    ("Sample convergence", "simulation_stability_convergence.csv"),
)
DATASET_CLASS_ORDER = ("normal", "hardware", "weather", "shading")


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _source_table(paper: Path, name: str, results: Optional[Path] = None) -> Path:
    if results is not None:
        current = results / "tables" / name
        if current.exists():
            return current
    active = paper / "tables" / name
    if active.exists():
        return active
    return paper / "supplementary" / "legacy_not_for_manuscript" / "tables" / name


def _number(value, digits: int = 3) -> str:
    if value is None or value == "":
        return "--"
    number = float(value)
    if not np.isfinite(number):
        return "--"
    return ("{:." + str(digits) + "f}").format(number)


def _escape_tex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _write_csv(path: Path, headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def _write_table(path: Path, label: str, caption: str, headers: Sequence[str],
                 rows: Sequence[Sequence[object]], long: bool = False,
                 group_rows: Sequence[int] = ()) -> None:
    if long:
        remaining = len(headers) - 1
        first_column_width = "2.5cm"
        columns = [
            r">{\raggedright\arraybackslash}p{" + first_column_width + "}"
        ]
        columns.extend(
            r">{\raggedright\arraybackslash}p{\dimexpr(\textwidth-"
            + first_column_width + "-" + str(2 * len(headers))
            + r"\tabcolsep)/"
            + str(remaining) + r"\relax}"
            for _ in range(remaining)
        )
        begin = "\\footnotesize\n\\begin{longtable}{" + "".join(columns) + "}\n"
        title = "\\caption{" + caption + "}\\\\\n\\label{" + label + "}\\\\\n"
        top = "\\toprule\n" + " & ".join(headers) + " \\\\\n\\midrule\n\\endfirsthead\n"
        repeat = "\\toprule\n" + " & ".join(headers) + " \\\\\n\\midrule\n\\endhead\n"
        body = []
        for index, row in enumerate(rows):
            if index in group_rows:
                body.append("\\addlinespace\n")
            body.append(" & ".join(_escape_tex(value) for value in row) + " \\\\\n")
        ending = "\\bottomrule\n\\end{longtable}\n\\normalsize\n"
        path.write_text(begin + title + top + repeat + "".join(body) + ending,
                        encoding="utf-8")
        return
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\small",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        "\\toprule",
        " & ".join(headers) + r" \\",
        "\\midrule",
    ]
    for index, row in enumerate(rows):
        if index in group_rows:
            lines.append("\\addlinespace")
        lines.append(" & ".join(_escape_tex(value) for value in row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_figure(fig, output: Path, name: str) -> None:
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(output / (name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.45,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _draw_simulation_injection_examples(paper: Path) -> None:
    from src.simulation.model import TrueParameters, generate_stationary_samples

    n_points = 2400
    normal_params = TrueParameters()
    normal_g, normal_p = generate_stationary_samples(20261011, n_points, normal_params)
    hardware_params = TrueParameters(alpha=8.0, beta=2.0, eta=0.52)
    hardware_g, hardware_p = generate_stationary_samples(
        20261011, n_points, hardware_params
    )
    hardware_rng = np.random.default_rng(20261030)
    spike_index = int(hardware_rng.integers(n_points))
    hardware_p[spike_index] += 8.0 * (
        hardware_params.sigma0 * hardware_params.eta * hardware_g[spike_index]
        + hardware_params.sigma_min
    )
    weather_g, weather_p = generate_stationary_samples(
        20261011, n_points,
        TrueParameters(alpha=4.0, beta=5.0),
    )
    shading_g = normal_g.copy()
    shading_p = normal_p.copy()
    shading_length = int(round(0.08 * n_points))
    shading_start = int(round(0.55 * n_points))
    shading_p[shading_start:shading_start + shading_length] *= 0.15

    cases = (
        ("Hardware anomaly: lower conversion slope + spike",
         normal_g, hardware_g, hardware_p, spike_index, "power"),
        ("Weather anomaly: changed irradiance distribution",
         normal_g, weather_g, weather_p, None, "weather"),
        ("Shading anomaly: power drop, irradiance unchanged",
         normal_g, shading_g, shading_p, (shading_start, shading_start + shading_length),
         "shading"),
    )
    colors = {"power": "#D95F02", "weather": "#2878B5", "shading": "#009E73"}
    ticks = np.arange(n_points)
    plot_indices = np.unique(np.r_[
        np.arange(0, n_points, 6), spike_index, shading_start,
        shading_start + shading_length - 1,
    ])
    _set_style()
    fig, axes = plt.subplots(3, 2, figsize=(10.2, 6.0), sharex=True)
    for row, (title, base_g, changed_g, changed_p, marker, kind) in enumerate(cases):
        input_ax, power_ax = axes[row]
        input_ax.plot(ticks[plot_indices], base_g[plot_indices],
                      color="#8C969F", linewidth=0.65,
                      label="Healthy simulation")
        input_ax.plot(ticks[plot_indices], changed_g[plot_indices],
                      color=colors[kind], linewidth=0.75,
                      alpha=0.9, label="Injected simulation")
        power_ax.plot(ticks[plot_indices], normal_p[plot_indices],
                      color="#8C969F", linewidth=0.65,
                      label="Healthy simulation")
        power_ax.plot(ticks[plot_indices], changed_p[plot_indices],
                      color=colors[kind], linewidth=0.75,
                      alpha=0.9, label="Injected simulation")
        if kind == "power" and marker is not None:
            power_ax.axvline(marker, color=colors[kind], linestyle="--", linewidth=0.8)
            power_ax.scatter(
                [marker], [changed_p[marker]], color=colors[kind], s=20,
                zorder=4, label="Injected spike",
            )
        elif kind == "shading" and marker is not None:
            power_ax.axvspan(marker[0], marker[1], color=colors[kind], alpha=0.12)
        input_ax.set_ylabel("Irradiance G")
        power_ax.set_ylabel("Power P")
        input_ax.set_title(title, loc="left", fontsize=8)
        input_ax.legend(frameon=False, fontsize=6, loc="upper right")
        power_ax.legend(frameon=False, fontsize=6, loc="upper right")
        input_ax.grid(True, alpha=0.2)
        power_ax.grid(True, alpha=0.2)
    axes[-1, 0].set_xlabel("Sample index")
    axes[-1, 1].set_xlabel("Sample index")
    fig.suptitle("Representative simulations of the three physically defined anomaly types")
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.0, w_pad=0.8)
    _save_figure(fig, paper / "figures", "sim_injection_examples")


def _draw_simulation_interpretability(paper: Path, results: Path) -> None:
    payload = _read_json(results / "simulation" / "logs" / "three_class_parameter_modes.json")
    shifts = payload["simulation"]["parameter_shifts"]
    scale = np.asarray(payload["simulation"]["normal_scale"], dtype=float)
    classes = ("hardware", "weather", "shading")
    matrix = np.asarray([
        [shifts[name][parameter] / scale[index]
         for index, parameter in enumerate(PARAMETERS)]
        for name in classes
    ])
    limit = max(1.0, float(np.max(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(6.8, 2.35))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(PARAMETERS)), PARAMETER_LABELS)
    ax.set_yticks(range(len(classes)), [CLASS_LABELS[name] for name in classes])
    ax.set_title("Simulation-derived parameter changes")
    ax.set_xlabel("Estimated parameter")
    ax.set_ylabel("Injected physical pathway")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, "{:+.2f}".format(matrix[row, col]),
                    ha="center", va="center", fontsize=7)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Shift / healthy scale")
    fig.tight_layout()
    _save_figure(fig, paper / "figures", "sim_interpretability")


def _draw_detection_timeliness(paper: Path, results: Path) -> None:
    from matplotlib.patches import Patch

    simulation_dir = results / "simulation" / "tables"
    injections = _read_csv(simulation_dir / "simulation_injection_summary.csv")
    timeliness = _read_csv(simulation_dir / "simulation_timeliness_summary.csv")
    confusion = _read_csv(
        results / "simulation" / "tables" / "three_class_simulation_confusion.csv"
    )
    standalone = [row for row in injections if row["strategy"] != "mixed"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7))

    ax = axes[0]
    groups = ("spike", "shading")
    positions = np.arange(3)
    width = 0.34
    colors = ("#2878B5", "#E1812C")
    for group_index, kind in enumerate(groups):
        rows = [
            next(row for row in standalone
                 if row["fault_type"] == kind and row["level"] == level)
            for level in ("low", "medium", "high")
        ]
        x = positions + (group_index - 0.5) * width
        rates = [float(row["detection_rate_mean"]) for row in rows]
        delays = [float(row["delay_mean_ticks"] or 0) for row in rows]
        ax.bar(x, rates, width=width, color=colors[group_index],
               label="Impulse" if kind == "spike" else "Sustained attenuation")
        for xpos, rate, delay in zip(x, rates, delays):
            ax.annotate("{:.0%}\n{:.1f} min".format(rate, delay),
                        (xpos, rate), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(positions, ("Low", "Medium", "High"))
    ax.set_ylim(0, 1.30)
    ax.set_ylabel("Event detection rate")
    ax.set_title("Injection response\n(bar: rate; label: mean delay)")
    bar_handles = [
        Patch(facecolor=colors[0], label="Bars: one-tick impulse"),
        Patch(facecolor=colors[1], label="Bars: sustained attenuation"),
    ]

    ax = axes[2]
    horizons = (1, 5, 10, 20, 50)
    timeliness_names = {
        "spike": "Impulse",
        "shading": "Sustained attenuation",
        "cloudy": "Irradiance change",
        "low_value": "Low output",
    }
    for row in timeliness:
        values = [float(row["dtm_" + str(horizon)]) for horizon in horizons]
        ax.plot(horizons, values, marker="o", markersize=3,
                label=timeliness_names.get(row["anomaly_type"], row["anomaly_type"]))
    ax.set_xscale("log")
    ax.set_xticks(horizons, [str(value) for value in horizons])
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Detection-time horizon (minutes)")
    ax.set_ylabel("Fraction detected within horizon")
    ax.set_title("Detection timeliness\n(N=900 per condition)")
    line_handles, line_labels = axes[2].get_legend_handles_labels()

    ax = axes[1]
    classes = ("hardware", "weather", "shading")
    matrix = np.asarray([
        [float(row[name]) for name in classes] for row in confusion
    ])
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(classes)), ["Hardware", "Weather", "Shading"])
    ax.set_yticks(range(len(classes)), ["Hardware", "Weather", "Shading"])
    ax.set_xlabel("Attributed type")
    ax.set_ylabel("Injected type")
    ax.set_title("Attribution on held-out\nsimulated samples")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            count = matrix[row_index, col_index]
            ax.text(col_index, row_index, "{:.0f}".format(count),
                    ha="center", va="center", fontsize=6.5,
                    color="white" if count > matrix.max() * 0.6 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
    fig.legend(
        handles=bar_handles + line_handles,
        labels=[handle.get_label() for handle in bar_handles] + line_labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.19), ncol=3,
        frameon=False, fontsize=5.5,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.39, top=0.80, wspace=0.58)
    _save_figure(fig, paper / "figures", "sim_results")


def _draw_simulation_stability(paper: Path, results: Path) -> None:
    """Combined parameter-estimation stability figure (five simulation analyses)."""
    table_dir = results / "simulation" / "tables"
    frames = {
        name: _read_csv(table_dir / ("simulation_stability_" + name + ".csv"))
        for name in ("bias_variance", "window", "noise", "contamination", "convergence")
    }
    parameters = ("alpha", "beta", "eta", "sigma0", "sigma_min")
    _set_style()
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.4))

    indexed = {row["parameter"]: row for row in frames["bias_variance"]}
    positions = np.arange(len(parameters))
    axes[0, 0].bar(positions - 0.2, [float(indexed[p]["bias"]) for p in parameters],
                   width=0.4, color="#2878B5", label="Bias")
    axes[0, 0].bar(positions + 0.2, [float(indexed[p]["variance"]) for p in parameters],
                   width=0.4, color="#E1812C", label="Variance")
    axes[0, 0].axhline(0, color="#555555", linewidth=0.7)
    axes[0, 0].set_xticks(positions, parameters, rotation=20)
    axes[0, 0].set_yscale("symlog", linthresh=1e-4)
    axes[0, 0].set_title("Bias and variance (L=30)", fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=6)

    window = frames["window"]
    for index, parameter in enumerate(parameters):
        rows = sorted((r for r in window if r["parameter"] == parameter),
                      key=lambda r: float(r["condition"]))
        axes[0, 1].plot([float(r["condition"]) for r in rows],
                        [float(r["mse"]) for r in rows],
                        marker="o", markersize=2.5, label=parameter)
    axes[0, 1].set_xlabel("Window length L (samples)", fontsize=7)
    axes[0, 1].set_ylabel("MSE", fontsize=7)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Window length", fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=5.5, ncol=2)

    noise = frames["noise"]
    for parameter in ("eta", "sigma0", "sigma_min"):
        rows = sorted((r for r in noise if r["parameter"] == parameter),
                      key=lambda r: float(r["condition"]))
        axes[0, 2].plot([float(r["condition"]) for r in rows],
                        [float(r["mse"]) for r in rows],
                        marker="o", markersize=2.5, label=parameter)
    axes[0, 2].set_xlabel("Measurement noise σ", fontsize=7)
    axes[0, 2].set_ylabel("MSE", fontsize=7)
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("Measurement noise", fontsize=8)
    axes[0, 2].legend(frameon=False, fontsize=6)

    contamination = frames["contamination"]
    for parameter in ("eta", "sigma0", "alpha"):
        rows = sorted((r for r in contamination if r["parameter"] == parameter),
                      key=lambda r: float(r["condition"]))
        axes[1, 0].plot([float(r["condition"]) for r in rows],
                        [float(r["bias"]) for r in rows],
                        marker="o", markersize=2.5, label=parameter)
    axes[1, 0].axhline(0, color="#555555", linestyle="--", linewidth=0.7)
    axes[1, 0].set_xlabel("Injected spike (σ)", fontsize=7)
    axes[1, 0].set_ylabel("Bias", fontsize=7)
    axes[1, 0].set_title("Single-point contamination", fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=6)

    convergence = sorted(frames["convergence"], key=lambda r: float(r["condition"]))
    counts = [float(r["condition"]) for r in convergence]
    axes[1, 1].plot(counts, [float(r["bias"]) for r in convergence],
                    marker="o", markersize=2.5, color="#2878B5", label="Bias(η)")
    axes[1, 1].axhline(0, color="#2878B5", linestyle="--", linewidth=0.7)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("Sample count N", fontsize=7)
    axes[1, 1].set_ylabel("Bias(η)", fontsize=7, color="#2878B5")
    twin = axes[1, 1].twinx()
    twin.plot(counts, [float(r["variance"]) for r in convergence],
              marker="s", markersize=2.5, color="#E1812C", label="Var(η)")
    twin.set_yscale("log")
    twin.set_ylabel("Var(η)", fontsize=7, color="#E1812C")
    axes[1, 1].set_title("Sample-size convergence", fontsize=8)

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0, 0.92,
        "Five simulation analyses, 100 independent\n"
        "replicates per condition:\n\n"
        "• bias / variance of the five estimates\n"
        "• window length L = 10…100\n"
        "• measurement noise σ = 0.01…0.2\n"
        "• single-point contamination (1/3/5 σ)\n"
        "• sample-count convergence N = 10…1000\n\n"
        "Power-side parameters stabilise from L ≥ 20;\n"
        "measurement noise is the dominant error source.",
        va="top", fontsize=6.5,
    )
    fig.suptitle("Parameter-estimation stability", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(fig, paper / "figures", "sim_stability")


def _draw_model_pipeline(paper: Path) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    _set_style()
    fig = plt.figure(figsize=(9.6, 3.15))
    grid = fig.add_gridspec(1, 2, width_ratios=(0.95, 1.65), wspace=0.25)
    ax_model = fig.add_subplot(grid[0, 0])
    rng = np.random.default_rng(20261011)
    irradiance = rng.beta(2.0, 2.5, 650)
    eta, sigma0, sigma_min = 0.85, 0.03, 0.005
    mean_power = np.minimum(eta * irradiance, 1.0)
    sigma = sigma0 * mean_power + sigma_min
    power = np.clip(mean_power + rng.normal(0, sigma), -0.05, 1.05)
    x = np.linspace(0, 1, 200)
    mu = np.minimum(eta * x, 1.0)
    sig = sigma0 * mu + sigma_min
    ax_model.scatter(irradiance, power, s=4, alpha=0.22, color="#2878B5",
                     label="Simulated observations")
    ax_model.plot(x, mu, color="#D95F02", lw=1.5, label=r"$\mu(g)$")
    ax_model.fill_between(x, mu - 2 * sig, mu + 2 * sig, color="#D95F02",
                          alpha=0.18, label=r"$\mu(g)\pm2\sigma(g)$")
    ax_model.set(xlim=(0, 1), ylim=(-0.06, 1.05),
                 xlabel="Normalised irradiance $g$",
                 ylabel="Normalised power $p$")
    ax_model.set_title("Conditional probability model")
    ax_model.legend(frameon=False, fontsize=6, loc="upper left")

    ax_flow = fig.add_subplot(grid[0, 1])
    ax_flow.set_xlim(0, 10)
    ax_flow.set_ylim(0, 6)
    ax_flow.axis("off")

    def box(x0, y0, width0, height0, text, color):
        patch = FancyBboxPatch(
            (x0, y0), width0, height0,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color, edgecolor="#425466", linewidth=0.8,
        )
        ax_flow.add_patch(patch)
        ax_flow.text(x0 + width0 / 2, y0 + height0 / 2, text,
                     ha="center", va="center", fontsize=7)

    def arrow(start, end):
        ax_flow.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=9, lw=0.8, color="#425466"
        ))

    box(0.1, 2.45, 1.65, 1.0, "Measured\n$G, P$", "#EAF2F8")
    box(2.2, 4.25, 2.45, 1.05,
        "Probability-model features\n(5 window parameters)", "#FFF2CC")
    box(2.2, 1.0, 2.45, 1.45,
        "5 residual features from\nexternal power reference", "#E2F0D9")
    box(5.2, 2.45, 2.0, 1.0, "Joint feature vector\n$\\mathbf{x}_t$", "#EAF2F8")
    box(7.75, 2.45, 2.0, 1.0, "Mahalanobis\ndistance rule", "#FCE4D6")
    arrow((1.75, 2.95), (2.2, 4.65))
    arrow((1.75, 2.95), (2.2, 1.72))
    arrow((4.65, 4.65), (5.2, 3.05))
    arrow((4.65, 1.72), (5.2, 2.8))
    arrow((7.2, 2.95), (7.75, 2.95))
    ax_flow.set_title("TEPAD: joint Mahalanobis decision", pad=5)
    fig.suptitle("Model schematic and detection method", y=1.01, fontsize=10)
    fig.tight_layout()
    _save_figure(fig, paper / "figures", "dat_model_pipeline")


def _robust_standardize(values: np.ndarray, normal_values: np.ndarray) -> np.ndarray:
    center = float(np.median(normal_values))
    scale = float(1.4826 * np.median(np.abs(normal_values - center)))
    if scale <= 1e-8:
        scale = float(np.std(normal_values, ddof=1)) if len(normal_values) > 1 else 0.0
    if scale <= 1e-8:
        scale = max(abs(center) * 1e-6, 1e-6)
    return (values - center) / scale


def _draw_dataset_interpretability(paper: Path, results: Path) -> None:
    data_path = (
        results / "simulation" / "tables"
        / "three_class_dataset_parameter_distributions.csv"
    )
    records = _read_csv(data_path)
    figure_rows = []
    for dataset in ("ding", "ds226", "lareunion"):
        dataset_rows = [row for row in records if row["dataset"] == dataset]
        normal_rows = [row for row in dataset_rows if row["class"] == "normal"]
        if not normal_rows:
            raise ValueError("normal observations missing for dataset {!r}".format(dataset))
        for group in DATASET_CLASS_ORDER:
            group_rows = [row for row in dataset_rows if row["class"] == group]
            if not group_rows:
                continue
            standardized = {}
            for parameter in PARAMETERS:
                standard_values = _robust_standardize(
                    np.asarray([float(row[parameter]) for row in group_rows]),
                    np.asarray([float(row[parameter]) for row in normal_rows]),
                )
                standardized[parameter] = standard_values
            figure_rows.append({
                "dataset": dataset,
                "class": group,
                "label": "{} · {}".format(DATASET_LABELS[dataset], group),
                "values": standardized,
            })

    fig, axes = plt.subplots(1, len(PARAMETERS), figsize=(10.2, 4.2), sharey=True)
    colors = {"normal": "#8C969F", "hardware": "#D55E00",
              "weather": "#0072B2", "shading": "#009E73"}
    class_display = {
        "normal": "Normal curves",
        "hardware": "Hardware anomaly (power conversion)",
        "weather": "Weather anomaly (irradiance input)",
        "shading": "Shading anomaly (module reception)",
    }
    row_positions = np.arange(1, len(figure_rows) + 1)
    for axis_index, parameter in enumerate(PARAMETERS):
        ax = axes[axis_index]
        for position, row in zip(row_positions, figure_rows):
            values = row["values"][parameter]
            bp = ax.boxplot(
                values, positions=[position], vert=False, widths=0.55,
                patch_artist=True, showfliers=False, manage_ticks=False,
                medianprops={"color": "#222222", "linewidth": 0.9},
                whiskerprops={"linewidth": 0.65},
                capprops={"linewidth": 0.65},
            )
            bp["boxes"][0].set_facecolor(colors[row["class"]])
            bp["boxes"][0].set_alpha(0.66)
        ax.axvline(0.0, color="#555555", linestyle="--", linewidth=0.7)
        ax.set_title(PARAMETER_LABELS[axis_index])
        if parameter == "sigma0":
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_xticks([-1e4, 0, 1e5])
            ax.set_xticklabels([r"$-10^4$", "0", r"$10^5$"], fontsize=5.5)
        ax.grid(axis="x")
    axes[0].set_yticks(row_positions, [row["label"] for row in figure_rows])
    axes[0].tick_params(axis="y", labelsize=6)
    legend_handles = [
        plt.Line2D([0], [0], color=colors[name], lw=5,
                   label=class_display[name])
        for name in DATASET_CLASS_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.55, -0.01))
    fig.suptitle("Estimated-parameter interpretability by dataset and mapped class", y=1.01)
    fig.supxlabel(
        "Parameter value relative to dataset-specific normal curves",
        y=0.07, fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.98), w_pad=0.8)
    _save_figure(fig, paper / "figures", "dat_parameter_distribution")


def _write_sim_interpretability(paper: Path, results: Path) -> None:
    payload = _read_json(results / "simulation" / "logs" / "three_class_parameter_modes.json")
    shifts = payload["simulation"]["parameter_shifts"]
    headers = ["injection", *PARAMETERS]
    rows = [
        [CLASS_LABELS[group], *[_number(shifts[group][name], 4) for name in PARAMETERS]]
        for group in ("hardware", "weather", "shading")
    ]
    table_dir = paper / "tables"
    _write_csv(table_dir / "sim_interpretability.csv", headers, rows)
    _write_table(
        table_dir / "sim_interpretability.tex", "tab:sim_interpretability",
        "Simulation-based interpretability analysis. Hardware anomaly means a power-conversion "
        "event that lowers efficiency or produces a power spike; weather anomaly means an "
        "irradiance-input event that changes irradiance distribution; shading anomaly means a "
        "module-reception event that lowers power while irradiance remains unchanged. Entries "
        "are median parameter shifts from the healthy simulated group.",
        ["Injected pathway", *PARAMETER_LABELS], rows,
        long=True,
    )


def _write_sim_detection(paper: Path, results: Path) -> None:
    records = _read_csv(results / "simulation" / "tables" / "simulation_injection_summary.csv")
    selected = [row for row in records if row["strategy"] != "mixed"]
    rows = []
    csv_rows = []
    for row in selected:
        fault = ("One-tick impulse" if row["fault_type"] == "spike"
                 else "Sustained power attenuation")
        duration = "1" if row["duration_ticks"] in ("", None) else _number(row["duration_ticks"], 0)
        values = [
            fault, row["level"], duration, row["n_trials"],
            _number(float(row["detection_rate_mean"]) * 100, 1) + "%",
            _number(row["delay_mean_ticks"], 2),
            _number(row["control_false_alarm_rate"], 3),
        ]
        rows.append(values)
        csv_rows.append([
            fault, row["level"], duration, row["n_trials"],
            _number(float(row["detection_rate_mean"]), 4),
            _number(row["delay_mean_ticks"], 4),
            _number(row["control_false_alarm_rate"], 4),
        ])
    _write_csv(
        paper / "tables" / "sim_detection.csv",
        ["injection", "intensity", "duration_ticks", "n_trials",
         "detection_rate", "mean_delay_ticks", "control_false_alarm_rate"],
        csv_rows,
    )
    _write_table(
        paper / "tables" / "sim_detection.tex", "tab:sim_detection",
        "Fault-injection detection results (100 paired trials per row). Detection delay is "
        "the mean delay among detected trials. The sustained power attenuation represents "
        "shading anomaly, defined as a module-reception event that lowers power while irradiance "
        "remains unchanged.",
        ["Injection", "Intensity", "Duration (min)", "$n$", "Detected", "Mean delay (min)",
         "Control false alarm"], rows, long=True,
    )


def _write_sim_classification(paper: Path, results: Path) -> None:
    payload = _read_json(results / "simulation" / "logs" / "three_class_parameter_modes.json")
    simulation = payload["simulation"]
    matrix = np.asarray(simulation["confusion_matrix"], dtype=int)
    groups = ("hardware", "weather", "shading")
    rows = [
        [CLASS_LABELS[groups[row]], *[int(value) for value in matrix[row]],
         _number(simulation["recalls"][groups[row]], 3)]
        for row in range(3)
    ]
    _write_csv(
        paper / "tables" / "sim_classification.csv",
        ["actual_class", *groups, "recall"], rows,
    )
    _write_table(
        paper / "tables" / "sim_classification.tex", "tab:sim_classification",
        "Held-out simulation confusion matrix. Hardware anomaly means a power-conversion event "
        "that lowers efficiency or produces a power spike; weather anomaly means an irradiance-input "
        "event that changes irradiance distribution; shading anomaly means a module-reception "
        "event that lowers power while irradiance remains unchanged. Rows are injected classes; columns are predicted "
        "classes. Each row contains {} independent test simulations.".format(
            simulation["repeats_per_class"]
        ),
        ["Actual injected pathway", "Power conversion", "Irradiance input",
         "Module reception", "Recall"], rows, long=True,
    )


def _write_sim_timeliness(paper: Path, results: Path) -> None:
    records = _read_csv(results / "simulation" / "tables" / "simulation_timeliness_summary.csv")
    display = {
        "spike": "One-tick impulse",
        "shading": "Sustained power attenuation",
        "cloudy": "Irradiance fluctuation",
        "low_value": "Sustained low output",
    }
    rows, csv_rows = [], []
    for row in records:
        source_name = row["anomaly_type"]
        values = [
            display.get(source_name, source_name), row["n_trials"],
            _number(float(row["detection_rate"]) * 100, 1) + "%",
            _number(row["mtd_ticks"], 2),
            _number(row["dtm_1"], 3), _number(row["dtm_5"], 3),
            _number(row["dtm_10"], 3), _number(row["dtm_50"], 3),
        ]
        rows.append(values)
        csv_rows.append([
            display.get(source_name, source_name), row["n_trials"],
            _number(row["detection_rate"], 4), _number(row["mtd_ticks"], 4),
            *[_number(row["dtm_" + str(horizon)], 4) for horizon in (1, 5, 10, 50)],
        ])
    _write_csv(
        paper / "tables" / "sim_timeliness.csv",
        ["event", "n_trials", "detection_rate", "mtd_ticks",
         "dtm_1", "dtm_5", "dtm_10", "dtm_50"], csv_rows,
    )
    _write_table(
        paper / "tables" / "sim_timeliness.tex", "tab:sim_timeliness",
        "Detection rate, mean time to detection (MTD), and detection-time measure (DTM) "
        "for 900 simulated trials per event. DTM is the fraction detected within the stated "
        "time horizon; MTD is expressed in one-minute samples.",
        ["Simulated event", "$n$", "Detected", "MTD (min)",
         "DTM $\\leq$1 min", "DTM $\\leq$5 min", "DTM $\\leq$10 min", "DTM $\\leq$50 min"],
        rows, long=True,
    )


def _write_sim_detection_classification_timeliness(
    paper: Path, results: Path
) -> None:
    simulation_tables = results / "simulation" / "tables"
    injection_source = _read_csv(simulation_tables / "simulation_injection_summary.csv")
    injection_rows = injection_source
    classification = _read_csv(
        simulation_tables / "three_class_simulation_confusion.csv"
    )
    timeliness = _read_csv(
        simulation_tables / "simulation_timeliness_summary.csv"
    )
    csv_rows = []
    for row in injection_rows:
        if row["strategy"] == "spike":
           case = "One-tick impulse"
        elif row["strategy"] == "shading":
           case = "Sustained power attenuation"
        else:
           case = "Mixed: {}".format(
               "spike" if row["fault_type"] == "spike" else "attenuation"
           )
        csv_rows.append([
           "Injection", case, row["level"], row["duration_ticks"] or "1",
           row["n_trials"], _number(row["detection_rate_mean"], 4),
           _number(row["delay_mean_ticks"], 4),
           _number(row["control_false_alarm_rate"], 4), _number(row["f1"], 4),
           "", "", "", "", "", "",
        ])
    for row in classification:
        csv_rows.append([
           "Classification", row["actual_class"], "", "", sum(
               int(row[name]) for name in ("hardware", "weather", "shading")
           ), "", "", "", "", _number(row["recall"], 4), "", "", "", "", "",
        ])
    for row in timeliness:
        csv_rows.append([
           "Timeliness", row["anomaly_type"], "", row["duration_ticks"],
           row["n_trials"], _number(row["detection_rate"], 4), "",
           _number(row["control_false_alarm_rate"], 4), "",
           "", _number(row["mtd_ticks"], 4),
           *[_number(row["dtm_" + str(horizon)], 4)
             for horizon in (1, 5, 10, 20, 50)],
        ])
    headers = [
        "section", "case", "severity", "duration_ticks", "n",
        "detection_rate", "mean_delay_ticks", "false_alarm_rate", "f1",
        "classification_recall", "mtd_ticks", "dtm_1", "dtm_5",
        "dtm_10", "dtm_20", "dtm_50",
    ]
    _write_csv(
        paper / "tables" / "sim_detection_classification_timeliness.csv",
        headers, csv_rows,
    )

    caption = (
        "Simulation results. Hardware anomaly means a power-conversion event that lowers "
        "conversion efficiency and includes a spike; weather anomaly means an irradiance-input "
        "event that changes irradiance distribution; shading anomaly means a module-reception "
        "event that lowers power while irradiance is unchanged. DTM is the fraction detected "
        "within the stated horizon; MTD is mean time to detection. Injection results and "
        "classification results use independent 100-trial samples per condition or class; "
        "timeliness uses 900 trials per condition."
    )
    lines = [
        "\\begin{table*}[!t]", "\\centering", "\\scriptsize",
        "\\caption{" + caption + "}",
        "\\label{tab:sim_detection_classification_timeliness}",
        "\\textit{Injection detection rate and delay (1 tick = 1 minute)}\\\\",
        "\\begin{tabular}{llrrrrrr}", "\\toprule",
        "Injected event & Severity & Duration & $n$ & Detection rate & "
        "Mean delay (ticks) & Control FPR & F1 \\\\", "\\midrule",
    ]
    for row in injection_rows:
        if row["strategy"] == "spike":
           event = "One-tick impulse"
        elif row["strategy"] == "shading":
           event = "Sustained attenuation"
        else:
           event = "Mixed spike" if row["fault_type"] == "spike" \
               else "Mixed attenuation"
        duration = row["duration_ticks"] or "1"
        values = [
           event, row["level"], duration, row["n_trials"],
           _number(float(row["detection_rate_mean"]) * 100, 1) + "%",
           _number(row["delay_mean_ticks"], 2),
           _number(row["control_false_alarm_rate"], 3), _number(row["f1"], 3),
        ]
        lines.append(" & ".join(_escape_tex(value) for value in values) + r" \\")
    lines.extend([
        "\\bottomrule", "\\end{tabular}\\\\[4pt]",
        "\\textit{Held-out classification confusion matrix and recall}\\\\",
        "\\begin{tabular}{lrrrr}", "\\toprule",
        "Actual injected class & Power conversion & Irradiance input & "
        "Module reception & Recall \\\\", "\\midrule",
    ])
    class_names = {
        "hardware": "Hardware anomaly",
        "weather": "Weather anomaly",
        "shading": "Shading anomaly",
    }
    for row in classification:
        values = [
           class_names[row["actual_class"]], row["hardware"], row["weather"],
           row["shading"], _number(row["recall"], 3),
        ]
        lines.append(" & ".join(_escape_tex(value) for value in values) + r" \\")
    lines.extend([
        "\\bottomrule", "\\end{tabular}\\\\[4pt]",
        "\\textit{Detection rate, mean detection time and detection-time measure}\\\\",
        "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
        "Injected event & $n$ & Detection rate & MTD (ticks) & "
        "DTM 1 & DTM 5 & DTM 10 & DTM 20 & DTM 50 \\\\", "\\midrule",
    ])
    event_names = {
        "spike": "One-tick impulse",
        "shading": "Sustained attenuation",
        "cloudy": "Irradiance fluctuation",
        "low_value": "Sustained low output",
    }
    for row in timeliness:
        values = [
           event_names[row["anomaly_type"]], row["n_trials"],
           _number(float(row["detection_rate"]) * 100, 1) + "%",
           _number(row["mtd_ticks"], 2),
           *[_number(row["dtm_" + str(horizon)], 3)
             for horizon in (1, 5, 10, 20, 50)],
        ]
        lines.append(" & ".join(_escape_tex(value) for value in values) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (paper / "tables" / "sim_detection_classification_timeliness.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_sim_stability(paper: Path, results: Path) -> None:
    rows = []
    csv_rows = []
    for analysis, filename in STABILITY_FILES:
        records = _read_csv(results / "simulation" / "tables" / filename)
        for record in records:
            condition = record["condition"]
            row = [
                analysis, condition, record["parameter"],
                _number(record["bias"], 4), _number(record["variance"], 5),
                _number(record["mse"], 5), record["n_repeats"],
            ]
            rows.append(row)
            csv_rows.append(row)
    _write_csv(
        paper / "tables" / "sim_stability.csv",
        ["analysis", "condition", "parameter", "bias", "variance", "mse", "n_repeats"],
        csv_rows,
    )
    starts = []
    previous = None
    for index, row in enumerate(rows):
        if row[0] != previous:
            starts.append(index)
            previous = row[0]
    _write_table(
        paper / "tables" / "sim_stability.tex", "tab:sim_stability",
        "Parameter-estimation stability across five simulation analyses. Bias, variance and "
        "mean-squared error (MSE) are computed from 100 independent replicates per condition.",
        ["Analysis", "Condition", "Parameter", "Bias", "Variance", "MSE", "$n$"],
        rows, long=True, group_rows=starts,
    )


def _write_dat_main(paper: Path, results: Path) -> None:
    main = _read_csv(_source_table(paper, "table1_main_ding.csv", results))
    classes = _read_csv(_source_table(paper, "table3_per_class_recall.csv", results))
    mapped_recalls = _read_csv(
        results / "simulation" / "tables" / "three_class_dataset_recalls.csv"
    )
    method_names = {
        "stage1": "Probability-model features only",
        "cascade": "Cascade (Stage 1 $\\cup$ Stage 2)",
    }
    dataset_names = {
        "ding": "Dataset A",
        "ds226": "Dataset B",
        "lareunion": "Dataset C",
    }
    class_names = {
        "hardware": "Hardware anomaly",
        "weather": "Weather anomaly",
        "shading": "Shading anomaly",
    }
    main_rows = [[
        method_names[row["method"]], "--" if not row["auc"] else _number(row["auc"]),
        _number(row["precision"]), _number(row["recall"]), _number(row["f1"]),
    ] for row in main]
    class_rows = [[
        row["cls"].replace("_", " "), row["n"],
        _number(row["stage1"]), _number(row["stage2"]), _number(row["cascade"]),
    ] for row in classes]
    mapped_rows = [[
        dataset_names[row["dataset"]],
        class_names[row["class"]],
        row["n"],
        "--" if not row["recall"] else _number(row["recall"]),
    ] for row in mapped_recalls]
    tex = [
        "\\begin{table*}[!t]", "\\centering", "\\small",
        "\\caption{Main results and per-label recall on Dataset A. Hardware anomaly means a "
        "power-conversion event that lowers efficiency or produces a power spike; weather anomaly "
        "means an irradiance-input event that changes irradiance distribution; shading anomaly "
        "means a module-reception event that lowers power while irradiance remains unchanged. "
        "Single-group columns are diagnostics; only joint Mahalanobis is the TEPAD decision.}",
        "\\label{tab:dat_main}",
        "\\textit{Overall performance}\\\\",
        "\\begin{tabular}{lrrrr}", "\\toprule",
        "Method & AUC & Precision & Recall & F1 \\\\", "\\midrule",
    ]
    tex.extend(" & ".join(_escape_tex(value) for value in row) + r" \\" for row in main_rows)
    tex.extend([
        "\\bottomrule", "\\end{tabular}\\\\[5pt]",
        "\\textit{Per-class recall}\\\\",
        "\\begin{tabular}{lrrrr}", "\\toprule",
        "Dataset A label & $n$ & Probability-model only & Reference residuals only & "
        "Joint Mahalanobis \\\\",
        "\\midrule",
    ])
    tex.extend(" & ".join(_escape_tex(value) for value in row) + r" \\" for row in class_rows)
    tex.extend([
        "\\bottomrule", "\\end{tabular}\\\\[5pt]",
        "\\textit{Recall using the simulation-trained attribution rule}\\\\",
        "\\begin{tabular}{llrr}", "\\toprule",
        "Dataset & Physically defined anomaly type & $n$ & Recall \\\\",
        "\\midrule",
    ])
    tex.extend(" & ".join(_escape_tex(value) for value in row) + r" \\" for row in mapped_rows)
    tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (paper / "tables" / "dat_main.tex").write_text("\n".join(tex), encoding="utf-8")
    _write_csv(
        paper / "tables" / "dat_main.csv",
        ["section", "item", "n", "auc", "precision", "recall", "f1",
         "probability_model_recall", "residual_feature_recall", "combined_recall"],
        [
            ["overall", row[0], "", row[1], row[2], row[3], row[4], "", "", ""]
            for row in main_rows
        ] + [
            ["per-class", row[0], row[1], "", "", "", "", row[2], row[3], row[4]]
            for row in class_rows
        ] + [
            ["mapped-attribution-recall", "{}: {}".format(row[0], row[1]),
             row[2], "", "", row[3], "", "", "", ""]
            for row in mapped_rows
        ],
    )


def _write_dat_cross_dataset(paper: Path, results: Path) -> None:
    cross = _read_csv(_source_table(paper, "table2_cross_dataset.csv", results))
    ablation = {
        row["dataset"]: row
        for row in _read_csv(_source_table(paper, "table4_ablation.csv", results))
    }
    rows = []
    csv_rows = []
    for row in cross:
        label = row["dataset"]
        change = ablation.get(label, {})
        values = [
            label, row["n_curves"], _number(row["stage1_auc"]),
            _number(row["precision"]), _number(row["recall"]), _number(row["f1"]),
            _number(change.get("delta_f1")),
        ]
        rows.append(values)
        csv_rows.append(values)
    _write_csv(
        paper / "tables" / "dat_cross_dataset.csv",
        ["dataset", "n_curves", "stage1_auc", "precision",
         "recall", "cascade_f1", "f1_gain_over_stage1"], csv_rows,
    )
    _write_table(
        paper / "tables" / "dat_cross_dataset.tex", "tab:dat_cross_dataset",
        "Cross-dataset results. Precision, recall and F1 refer to the cascade decision; "
        "the AUC column is the Stage-1 probability-model Mahalanobis ranking score.",
        ["Dataset", "Curves", "Stage-1 AUC", "Precision", "Recall",
         "Cascade F1", "$\\Delta$F1"], rows, long=True,
    )


def _write_dat_baseline(paper: Path, results: Path) -> None:
    ding = _read_csv(_source_table(paper, "table6_model_comparison.csv", results))
    rows = []
    for row in ding:
        if row.get("error"):
            continue
        rows.append([
            "Dataset A", "Held-out", row["name"],
            _number(row["auc"]), _number(row["precision"]), _number(row["recall"]),
            _number(row["f1"]), _number(row["fpr"]),
        ])
    _write_csv(
        paper / "tables" / "dat_baseline.csv",
        ["dataset", "evaluation_split", "method", "auc", "precision",
         "recall", "f1", "false_positive_rate"], rows,
    )
    _write_table(
        paper / "tables" / "dat_baseline.tex", "tab:dat_baseline",
        "Baseline comparison on Dataset A using a fixed train/test split. TEPAD additionally "
        "uses an external power reference; all other methods use measured irradiance and power "
        "only. The annotated supervised baseline uses training labels.",
        ["Dataset", "Protocol", "Method", "AUC", "Precision", "Recall", "F1", "FPR"],
        rows, long=True,
    )


def _write_dat_interpretability(paper: Path, results: Path) -> None:
    records = _read_csv(
        results / "simulation" / "tables"
        / "three_class_dataset_parameter_distributions.csv"
    )
    groups = defaultdict(list)
    for row in records:
        groups[(row["dataset"], row["class"])].append(row)
    class_names = {
        "normal": "Normal",
        "hardware": "Hardware",
        "weather": "Weather",
        "shading": "Shading",
        "unmapped": "Unmapped label",
    }
    rows, csv_rows = [], []
    for dataset in ("ding", "ds226", "lareunion"):
        for group in DATASET_CLASS_ORDER:
            values = groups.get((dataset, group), [])
            if not values:
                continue
            summary = []
            for parameter in PARAMETERS:
                numbers = np.asarray([float(row[parameter]) for row in values])
                q1, median, q3 = np.percentile(numbers, (25, 50, 75))
                summary.append("{:.3g} [{:.3g}, {:.3g}]".format(median, q1, q3))
            row = [DATASET_LABELS[dataset], class_names[group], len(values), *summary]
            rows.append(row)
            csv_rows.append([dataset, group, len(values), *summary])
    _write_csv(
        paper / "tables" / "dat_parameter.csv",
        ["dataset", "class", "n", *PARAMETERS], csv_rows,
    )
    _write_table(
        paper / "tables" / "dat_parameter.tex", "tab:dat_parameter",
        "Dataset interpretability analysis of estimated parameters (median [25th, 75th percentiles]). "
        "Hardware anomaly "
        "means a power-conversion event that lowers efficiency or produces a power spike; weather "
        "anomaly means an irradiance-input event that changes irradiance distribution; shading "
        "anomaly means a module-reception event that lowers power while irradiance remains "
        "unchanged. Dataset labels are used only to evaluate these research-defined classes.",
        ["Dataset", "Mapped class", "$n$", *PARAMETER_LABELS],
        rows, long=True,
    )


def _archive_legacy_assets(paper: Path) -> None:
    archived_figures = set(FIGURES)
    archived_tables = set(TABLES)
    for directory_name, allowed_names, extensions in (
        ("figures", archived_figures, {".pdf", ".png"}),
        ("tables", archived_tables, {".tex", ".csv"}),
    ):
        source = paper / directory_name
        archive = paper / "supplementary" / "legacy_not_for_manuscript" / directory_name
        if not source.exists():
            continue
        archive.mkdir(parents=True, exist_ok=True)
        for path in list(source.iterdir()):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            if path.stem in allowed_names:
                continue
            destination = archive / path.name
            if destination.exists():
                destination = archive / "{}_{}{}".format(
                    path.stem, "previous", path.suffix
                )
            shutil.move(str(path), str(destination))


def _write_index(paper: Path) -> None:
    text = """# Paper figure and table index

The figures in the manifest have a matching PDF vector copy and PNG preview, and the data tables have matching TeX and CSV. The manuscript cites the PDF/TeX files; the CSV files are used to check traceable values. Files generated earlier that are not in this manifest have been moved to `supplementary/legacy_not_for_manuscript/`.

Research class definitions: hardware anomaly (anomaly in the power-conversion stage that lowers conversion efficiency or produces power spikes); weather anomaly (anomaly in the irradiance-input stage that changes the irradiance distribution); shading anomaly (anomaly in the module-receipt stage that lowers output power without changing the incident irradiance). Classes are defined by the physical stage of the PV system and by the simulation injections; dataset labels are used for validation only.

See [`paper_assets.md`](./paper_assets.md) for detailed conclusions, run values, physical interpretation and suggested locations in the manuscript.

## Figures

| File basename | Content |
|---|---|
| `sim_injection_examples` | Representative irradiance input and output power simulations for the three study-defined anomaly classes |
| `sim_interpretability` | Healthy-scale standardized plot of the simulated parameter changes |
| `sim_results` | Injection detection rate/delay, detection timeliness (DTM) and the three-class attribution confusion matrix |
| `sim_stability` | Parameter estimation stability across the five simulation analyses |
| `dat_model_pipeline` | Schematic of the TEPAD model and the joint criterion |
| `dat_parameter_distribution` | Parameter distributions of the mapped classes in the datasets |

> `tepadarch` and `2stage` are hand-drawn method diagrams of the paper, provided by `research_paper/paper/solar_energy/figures/`;
> this pipeline only keeps them and does not redraw them, so they have no corresponding CSV numeric source.

## Tables

| File basename | Content |
|---|---|
| `sim_interpretability` | Actual values of the five simulated parameter shifts |
| `sim_detection_classification_timeliness` | Injection detection rate/delay, simulation classification confusion matrix/recall and detection timeliness |
| `sim_stability` | Parameter estimation stability experiment |
| `dat_main` | Main experiment, per-class results on the original labels and recall of the physical-class mapping |
| `dat_cross_dataset` | Detection metrics of the three datasets |
| `dat_baseline` | Same-protocol comparison against baseline methods |
| `dat_parameter` | Five-parameter medians and quartiles of the mapped dataset classes |
"""
    (paper / "figure_table_index_and_caption_notes.md").write_text(text, encoding="utf-8")


def publish(paper: Path, results: Path) -> None:
    paper.mkdir(parents=True, exist_ok=True)
    figures_dir = paper / "figures"
    tables_dir = paper / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    required = [
        results / "simulation" / "logs" / "three_class_parameter_modes.json",
        results / "simulation" / "tables" / "three_class_dataset_parameter_distributions.csv",
        results / "simulation" / "tables" / "simulation_injection_summary.csv",
        results / "simulation" / "tables" / "three_class_simulation_confusion.csv",
        results / "simulation" / "tables" / "simulation_timeliness_summary.csv",
        results / "simulation" / "tables" / "simulation_stability_bias_variance.csv",
        results / "simulation" / "tables" / "three_class_dataset_recalls.csv",
        _source_table(paper, "table1_main_ding.csv", results),
        _source_table(paper, "table2_cross_dataset.csv", results),
        _source_table(paper, "table3_per_class_recall.csv", results),
        _source_table(paper, "table4_ablation.csv", results),
        _source_table(paper, "table6_model_comparison.csv", results),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required experiment artifacts are missing: {}".format(missing))

    _write_sim_interpretability(paper, results)
    _write_sim_detection_classification_timeliness(paper, results)
    _write_sim_stability(paper, results)
    _write_dat_main(paper, results)
    _write_dat_cross_dataset(paper, results)
    _write_dat_baseline(paper, results)
    _write_dat_interpretability(paper, results)

    _draw_simulation_injection_examples(paper)
    _draw_simulation_interpretability(paper, results)
    _draw_detection_timeliness(paper, results)
    _draw_simulation_stability(paper, results)
    _draw_model_pipeline(paper)
    _draw_dataset_interpretability(paper, results)
    _archive_legacy_assets(paper)
    _write_index(paper)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the curated manuscript figure and table set."
    )
    parser.add_argument("--paper", default=str(config.PAPER_DIR))
    parser.add_argument("--results", default=str(config.RESULTS_DIR))
    args = parser.parse_args()
    publish(Path(args.paper).resolve(), Path(args.results).resolve())
    print("Published {} figures and {} tables.".format(len(FIGURES), len(TABLES)))


if __name__ == "__main__":
    main()
