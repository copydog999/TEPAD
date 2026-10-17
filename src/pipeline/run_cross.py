# -*- coding: utf-8 -*-
"""Run the cross-dataset validation only: ``python src/pipeline/run_cross.py --output results/``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.pipeline.run_all import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="TEPAD cross-dataset validation (DS226 / La Réunion / SolarMagix)")
    ap.add_argument("--output", default="results/")
    args = ap.parse_args()
    config.ensure_dirs()
    out = Path(args.output)
    if not out.is_absolute():
        out = config.PROJECT_ROOT / out
    run(list(config.DATASETS_CROSS), out)


if __name__ == "__main__":
    main()
