# -*- coding: utf-8 -*-
"""Run the comparison experiment only: ``python src/pipeline/run_baselines.py --output results/``.

Outputs (same format as the main experiment results):

* ``results/tables/table6_model_comparison.{tex,csv}``
* ``results/figures/fig7_model_comparison.{pdf,png}``
* ``results/logs/baselines.json``
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.baselines.runner import MODELS, run_baselines  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="TEPAD comparison experiment (cross-model comparison on the Ding dataset)")
    ap.add_argument("--output", default="results/", help="output directory")
    ap.add_argument("--dataset", default="ding", help="dataset (only ding is currently supported)")
    ap.add_argument("--models", default="", help="comma-separated model subset, all by default; "
                                                 "available: " + ", ".join(MODELS))
    ap.add_argument("--epochs", type=int, default=config.BASELINE_EPOCHS,
                    help="number of training epochs for deep-learning baselines")
    ap.add_argument("--cache-dir", default="",
                    help="weight/score cache directory (default <output>/models)")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the cache and force retraining of deep models")
    args = ap.parse_args()

    config.ensure_dirs()
    out = Path(args.output)
    if not out.is_absolute():
        out = config.PROJECT_ROOT / out

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "logs" / "run_baselines.log",
                                                      encoding="utf-8")])
    models = [m for m in args.models.split(",") if m] or None
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    run_baselines(dataset=args.dataset, output=out, models=models, epochs=args.epochs,
                  cache_dir=cache_dir, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
