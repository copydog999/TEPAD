"""Run the simulation-derived parameter-shift classification experiment."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.evaluation.parameter_modes import (  # noqa: E402
    CLASS_ORDER,
    DATASET_LABEL_GROUPS,
    PARAMETER_NAMES,
    direction_checks,
    run_dataset_experiment,
    run_simulation_experiment,
)


_CLASS_DESCRIPTIONS = {
    "hardware": "hardware anomaly (anomaly in the power-conversion stage that lowers power-conversion efficiency and is accompanied by spikes)",
    "weather": "weather anomaly (anomaly in the irradiance-input stage that changes the shape of the irradiance distribution)",
    "shading": "shading anomaly (anomaly in the module-receipt stage that lowers output power while irradiance stays unchanged)",
}
_DATASET_NAMES = {
    "ding": "Ding dataset",
    "ds226": "DS226 dataset",
    "lareunion": "La Réunion dataset",
}
_DISPLAY_LABELS = {
    "ding": {
        "spike": "spike",
        "inverter fault": "inverter_fault",
        "low value": "low_value",
        "cloudy": "cloudy",
        "snowy": "snowy",
        "shading": "shading",
    },
    "ds226": {
        "degradation": "Degradation",
        "open circuit": "Open Circuit",
        "short circuit": "Short Circuit",
        "shading": "Shading",
    },
    "lareunion": {
        "1": "1.0",
        "2.1": "2.1",
        "2.2": "2.2",
        "2.3": "2.3",
        "3.1": "3.1",
        "3.2": "3.2",
        "4": "4.0",
    },
}


def _fmt(value: float) -> str:
    return "{:+.4f}".format(float(value))


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _label_mapping_lines() -> List[str]:
    lines = [
        "| Dataset | Original anomaly label | Category defined in this study |",
        "|---|---|---|",
    ]
    for dataset in ("ding", "ds226", "lareunion"):
        groups = DATASET_LABEL_GROUPS[dataset]
        by_group: Dict[str, List[str]] = {name: [] for name in CLASS_ORDER}
        for label, group in groups.items():
            by_group[group].append(_DISPLAY_LABELS[dataset][label])
        for group in CLASS_ORDER:
            labels = ", ".join(by_group[group]) if by_group[group] else "none"
            lines.append(
                "| {} | {} | {} |".format(
                    _DATASET_NAMES[dataset], labels, _CLASS_DESCRIPTIONS[group]
                )
            )
    lines.extend([
        "",
        "Normal labels are not scored as anomaly samples: `normal` for Ding, `Normal` for",
        "DS226 and `0.0` for La Réunion. Dataset anomaly labels are only used for validation",
        "after the simulation classifier has produced its predictions; they neither take part",
        "in the definition of the three classes above nor are used to fit the simulation class",
        "centers. The dataset parameter centers are the per-parameter median over all daily",
        "curves without reading labels; the residual reference slope of DS226 is likewise the",
        "median efficiency over all curves, without reading labels.",
        "",
    ])
    return lines


def _render_report(simulation: dict, datasets: List[dict]) -> str:
    shifts = simulation["parameter_shifts"]
    matrix = simulation["confusion_matrix"]
    checks = direction_checks(shifts)
    lines = [
        "# PV Experiment Report",
        "",
        "## 1. Theoretical Framework",
        "",
        "This study defines PV anomalies as three mutually exclusive validation classes "
        "according to the physical stage of the system:",
        "- **{}**: {}.".format("Hardware anomaly", _CLASS_DESCRIPTIONS["hardware"].split("(", 1)[1][:-1]),
        "- **{}**: {}.".format("Weather anomaly", _CLASS_DESCRIPTIONS["weather"].split("(", 1)[1][:-1]),
        "- **{}**: {}.".format("Shading anomaly", _CLASS_DESCRIPTIONS["shading"].split("(", 1)[1][:-1]),
        "",
        "The interpretability analysis in this section compares the statistical changes of",
        "the irradiance distribution, the power-conversion efficiency and the reference",
        "residual between the healthy simulation baseline and the three anomaly classes;",
        "anomaly-type attribution assigns anomaly samples to one of the three study-defined",
        "classes above using the parameter-change combinations established by the simulation.",
        "The class definitions come from the physical structure and the simulation settings",
        "of this experiment, not from any dataset label.",
        "",
        "Let the irradiance be $G$ and the output power be $P$; the estimated model",
        "parameters are: $\\alpha$ and $\\beta$ are the Beta shape parameters of the",
        "normalized irradiance distribution; $\\eta$ is the estimated conversion slope of",
        "$P$ against $G$; $\\sigma_0$ is the slope of the absolute residual scale as model",
        "power grows; $r_{\\min}$ is the most negative signed reference residual",
        "$P-P^{\\mathrm{ref}}$ in the curve, where more negative values mean a deeper",
        "minimum-power deviation. In the simulation $P^{\\mathrm{ref}}=0.82G$; the datasets",
        "prefer their existing external reference power and, when no such external reference",
        "exists, use the in-data median conversion efficiency computed without labels,",
        "multiplied by $G$.",
        "",
        "The three parameter combinations are built from the class centers of the simulation",
        "training set; at classification time the robust standardized offset of each",
        "parameter relative to the healthy simulation center is computed and the sample is",
        "assigned to the nearest class center in Euclidean distance. Training and test",
        "samples use non-overlapping random seeds; dataset labels do not participate in",
        "training, threshold selection or parameter-center computation.",
        "",
        "### Dataset label mapping (validation only)",
        "",
    ]
    lines.extend(_label_mapping_lines())

    lines.extend([
        "## 2. Simulation Interpretability Analysis",
        "",
        "The healthy baseline parameters are $(\\alpha,\\beta,\\eta,\\sigma_0,\\sigma_{\\min})"
        "=(8,2,0.82,0.025,0.008)$. The conversion-stage injection sets $\\eta$ to 0.52 and adds"
        " a spike of 8 local noise standard deviations at one random sample point; the"
        " irradiance-input injection changes the Beta parameters to $(4,5)$ while keeping the"
        " power relation unchanged; the module-receipt injection multiplies the power of a"
        " contiguous 8% of samples by 0.15 while the irradiance samples stay unchanged. Each"
        " repetition uses a different random seed.",
        "",
        "The table below reports, for each class, the median shift of every parameter relative"
        " to the median of the healthy test group in the independent test simulation. A plus"
        " sign means an increase and a minus sign a decrease; the direction signs are produced"
        " by the values of this actual run. Each class has {} test samples and each simulation"
        " curve has {} sample points.".format(
            simulation["repeats_per_class"], simulation["points_per_curve"]
        ),
        "",
        "| Simulation-injection class (study definition) | α shift | β shift | η shift | σ₀ shift | r_min shift |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for group in CLASS_ORDER:
        row = [shifts[group][name] for name in PARAMETER_NAMES]
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                _CLASS_DESCRIPTIONS[group], *(_fmt(value) for value in row)
            )
        )
    lines.extend([
        "",
        (
            "Direction-by-direction check: the conversion-stage injection lowers the estimated"
            " conversion efficiency: {}; the irradiance-input injection changes $\\alpha$: {} and"
            " changes $\\beta$: {}; the module-receipt injection lowers $r_{{\\min}}$ while"
            " keeping irradiance unchanged: {}. All of these judgments are computed from the"
            " parameter shifts of the same run shown in the table above."
        ).format(
            "pass" if checks["hardware_eta_decreases"] else "fail",
            "pass" if checks["weather_alpha_changes"] else "fail",
            "pass" if checks["weather_beta_changes"] else "fail",
            "pass" if checks["shading_r_min_decreases"] else "fail",
        ),
        "",
        "## 3. Simulation Three-Class Classification Results",
        "",
        "The three-class confusion matrix is computed on independent simulation test samples;",
        "rows are the true injection class and columns are the classifier output; the recall of",
        "each class is the number of correctly classified samples in that row divided by the",
        "row total.",
        "",
        "| True injection class (study definition) | Predicted power-conversion anomaly | Predicted irradiance-input anomaly | Predicted module-receipt anomaly |",
        "|---|---:|---:|---:|",
    ])
    for row_index, group in enumerate(CLASS_ORDER):
        lines.append(
            "| {} | {} | {} | {} |".format(
                _CLASS_DESCRIPTIONS[group],
                *[int(matrix[row_index, col]) for col in range(len(CLASS_ORDER))]
            )
        )
    lines.extend([
        "",
        "| Simulation-injection class (study definition) | Recall |",
        "|---|---:|",
    ])
    for group in CLASS_ORDER:
        lines.append("| {} | {:.3f} |".format(
            _CLASS_DESCRIPTIONS[group], simulation["recalls"][group]
        ))

    lines.extend([
        "",
        "## 4. Dataset Three-Class Classification Results",
        "",
        "Each recall counts only the anomaly daily curves mapped to the corresponding",
        "study-defined class. Each cell gives the recall and the number of annotated samples;",
        "no samples means the dataset's label set has no such class, and must not be read as",
        "zero recall.",
        "",
        "| Dataset | Power-conversion anomaly recall | Irradiance-input anomaly recall | Module-receipt anomaly recall |",
        "|---|---:|---:|---:|",
    ])
    for result in datasets:
        cells = []
        for group in CLASS_ORDER:
            count = result["class_counts"][group]
            recall = result["recalls"][group]
            cells.append("no samples" if recall is None else "{:.3f} (n={})".format(recall, count))
        lines.append("| {} | {} | {} | {} |".format(
            _DATASET_NAMES[result["dataset"]], *cells
        ))

    lines.extend([
        "",
        "## 5. Conclusion",
        "",
    ])
    if all(checks.values()):
        lines.append(
            "This simulation run verified the three configured physical-change directions: the "
            "power-conversion injection lowers $\\eta$, the irradiance-input injection changes "
            "$\\alpha$ and $\\beta$, and the module-receipt injection deepens the negative "
            "minimum reference residual without changing $G$. The simulation three-class recalls "
            "and the per-class recalls of the three real datasets are shown in the tables above; "
            "dataset labels are used for scoring only after prediction."
        )
    else:
        failed = [key for key, passed in checks.items() if not passed]
        lines.append(
            "This run did not satisfy all preset direction checks ({}), so it does not support "
            "claiming that all three parameter directions are verified by simulation. Rely on "
            "the measured values in the parameter-shift and recall tables.".format(", ".join(failed))
        )
    lines.extend([
        "",
        "Reproduction command: `python -m src.pipeline.run_parameter_modes`. The run seed, ",
        "simulation repeat counts, class label counts, matrices and recalls are also stored in ",
        "the adjacent log and table files.",
        "",
    ])
    return "\n".join(lines)


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the simulation-derived PV parameter-shift classification."
    )
    parser.add_argument("--output", default=None, help="output directory (default: results/)")
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--points", type=int, default=2400)
    args = parser.parse_args()

    output = Path(args.output) if args.output else config.RESULTS_DIR
    if not output.is_absolute():
        output = config.PROJECT_ROOT / output
    experiment_dir = output / "simulation"
    tables_dir = experiment_dir / "tables"
    logs_dir = experiment_dir / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    simulation = run_simulation_experiment(args.seed, args.repeats, args.points)
    checks = direction_checks(simulation["parameter_shifts"])
    if not all(checks.values()):
        raise RuntimeError("simulation did not verify all required parameter directions: {}"
                           .format(checks))
    classifier = simulation.pop("classifier")
    datasets = run_dataset_experiment(classifier)

    report_path = experiment_dir / "three_class_parameter_modes_report.md"
    report_path.write_text(_render_report(simulation, datasets), encoding="utf-8")

    deviation_rows = []
    for group in CLASS_ORDER:
        row = {"class": group}
        row.update({
            name: simulation["parameter_shifts"][group][name]
            for name in PARAMETER_NAMES
        })
        deviation_rows.append(row)
    _write_csv(
        tables_dir / "three_class_parameter_deviations.csv",
        deviation_rows,
        ["class"] + list(PARAMETER_NAMES),
    )

    confusion_rows = []
    matrix = simulation["confusion_matrix"]
    for i, actual in enumerate(CLASS_ORDER):
        row = {"actual_class": actual}
        row.update({pred: int(matrix[i, j]) for j, pred in enumerate(CLASS_ORDER)})
        row["recall"] = simulation["recalls"][actual]
        confusion_rows.append(row)
    _write_csv(
        tables_dir / "three_class_simulation_confusion.csv",
        confusion_rows,
        ["actual_class"] + list(CLASS_ORDER) + ["recall"],
    )

    recall_rows = []
    for result in datasets:
        for group in CLASS_ORDER:
            recall_rows.append({
                "dataset": result["dataset"],
                "class": group,
                "n": result["class_counts"][group],
                "recall": result["recalls"][group],
            })
    _write_csv(
        tables_dir / "three_class_dataset_recalls.csv",
        recall_rows,
        ["dataset", "class", "n", "recall"],
    )

    distribution_rows = [
        row
        for result in datasets
        for row in result["parameter_distributions"]
    ]
    _write_csv(
        tables_dir / "three_class_dataset_parameter_distributions.csv",
        distribution_rows,
        ["dataset", "label", "class"] + list(PARAMETER_NAMES),
    )

    log_data = {
        "simulation": simulation,
        "dataset_results": datasets,
        "label_groups": DATASET_LABEL_GROUPS,
        "direction_checks": checks,
    }
    (logs_dir / "three_class_parameter_modes.json").write_text(
        json.dumps(_jsonable(log_data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(    "Wrote {}".format(report_path))


if __name__ == "__main__":
    main()
