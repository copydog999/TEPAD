"""One-click reproduction of every experiment reported in the paper.

Usage
-----
    python scripts/reproduce.py                # full run (default N=100 simulation repeats)
    python scripts/reproduce.py --quick        # few repeats, layout/smoke check only
    python scripts/reproduce.py --skip-checks  # run everything, print numbers without failing

The script first checks the interpreter version and the required raw-data files,
then runs the paper's experiments in order and finally compares the produced
metrics with the values reported in ``research_paper/paper/paper.tex``, printing
every comparison. Differences beyond the tolerance exit non-zero; the measured
values are also written to ``results/logs/`` for independent inspection.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
DING_LABELS = ("normal", "spike", "inverter_fault", "snowy", "cloudy", "low_value", "shading")

#: Acceptance thresholds for the paper's main results: (Stage-1 AUC, cascade P, cascade R, cascade F1, tolerance)
EXPECTED_DATASETS: Dict[str, Tuple[float, float, float, float, float]] = {
    "ding": (0.908, 0.985, 0.997, 0.991, 0.002),
    "ds226": (1.000, 0.889, 1.000, 0.941, 0.002),
    "lareunion": (0.536, 0.548, 1.000, 0.708, 0.002),
}


def missing_inputs() -> List[Path]:
    """Return required raw-data files that are not present."""
    raw = ROOT / "datasets"
    required = [
        raw / "ds226" / "raw" / "PV_Diagnostic_Final_Labels.csv",
        raw / "lareunion" / "raw" / "dt1_solar_met.csv",
        raw / "lareunion" / "raw" / "dt2_inverter1_faults.csv",
        raw / "lareunion" / "raw" / "dt3_inverter2_normal.csv",
    ]
    for split in ("train", "test"):
        required.extend(
            raw / "ding" / "raw" / split / "{}_{}.csv".format(label, split)
            for label in DING_LABELS
        )
    return [path for path in required if not path.is_file()]


def run(label: str, *args: str) -> None:
    """Run a project command from the repository root and stop on failure."""
    print("\n=== {} ===".format(label), flush=True)
    subprocess.run([sys.executable, *args], cwd=str(ROOT), check=True)


def verify(output: Path) -> List[str]:
    """Compare measured metrics with the paper values; return the differences found."""
    metrics = json.loads((output / "logs" / "metrics.json").read_text(encoding="utf-8"))
    problems: List[str] = []
    print("\n=== Verification against the paper ===")
    for name, (auc, precision, recall, f1, tolerance) in EXPECTED_DATASETS.items():
        measured = metrics.get(name)
        if measured is None:
            problems.append("{}: missing from metrics.json".format(name))
            continue
        actual = (
            measured.get("stage1_auc"),
            measured.get("cascade_precision"),
            measured.get("cascade_recall"),
            measured.get("cascade_f1"),
        )
        print("  {:<10} AUC={:.3f} P={:.3f} R={:.3f} F1={:.3f}".format(name, *actual))
        for label, value, target in zip(
            ("stage1_auc", "cascade_precision", "cascade_recall", "cascade_f1"),
            actual, (auc, precision, recall, f1),
        ):
            if value is None or abs(float(value) - target) > tolerance:
                problems.append(
                    "{} {}: measured {} != paper {}".format(name, label, value, target)
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/", help="result root (default: results/)")
    parser.add_argument("--quick", action="store_true",
                        help="small repeat counts for a fast smoke run")
    parser.add_argument("--skip-checks", action="store_true",
                        help="do not fail when metrics differ from the paper")
    args = parser.parse_args()

    if sys.version_info < (3, 8):
        print(
            "TEPAD requires Python 3.8 or newer; current interpreter is {}. "
            "Activate a compatible environment and install requirements.txt.".format(
                sys.version.split()[0]
            ),
            file=sys.stderr,
        )
        return 2

    missing = missing_inputs()
    if missing:
        print("Required raw data is missing. Prepare the datasets first:", file=sys.stderr)
        for path in missing:
            print("  - {}".format(path.relative_to(ROOT)), file=sys.stderr)
        print(
            "\nRun 'python scripts/download_datasets.py --download' to fetch all three "
            "datasets from Zenodo (about 2.9 GB), or pass --source to lay them out from a "
            "local copy. See docs/datasets.md and docs/reproduction.md.",
            file=sys.stderr,
        )
        return 2

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    repeats = 5 if args.quick else 100
    if args.quick:
        # With few repeats the axis ranges differ and the compact layout may overlap text;
        # smoke runs only warn.
        os.environ["TEPAD_FIG_STRICT_OVERLAP"] = "0"

    run("Paper tables and figures: main, cross-dataset and baseline experiments",
        "src/pipeline/run_all.py", "--datasets", "all", "--baselines",
        "--output", args.output)
    run("Simulation: fault injection, interpretability, timeliness, stability",
        "src/pipeline/run_simulation.py", "--output", args.output,
        "--repeats", str(repeats), "--stability-repeats", str(repeats))
    run("Simulation: shading definitions and detection timeliness (DTM)",
        "src/pipeline/run_definitions.py", "--output", args.output,
        "--repeats", str(repeats), "--noise-repeats", str(repeats))
    run("Simulation: three-class parameter attribution",
        "src/pipeline/run_parameter_modes.py", "--output", args.output,
        "--repeats", str(repeats))
    run("Publish the manuscript figures and tables",
        "src/pipeline/publish_paper_assets.py", "--results", args.output)

    problems = verify(output)
    if problems and not args.skip_checks:
        print("\nReproduction FAILED: measured metrics differ from the paper.", file=sys.stderr)
        for problem in problems:
            print("  - {}".format(problem), file=sys.stderr)
        return 1
    if problems:
        print("\nWARNING (checks skipped): {}".format("; ".join(problems)), file=sys.stderr)
    print("\nReproduction completed. Results are in {}.".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
