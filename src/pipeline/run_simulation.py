"""Run the repeated synthetic fault-injection and parameter-stability experiments."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.simulation.experiment import run_simulation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TEPAD simulated fault injection and estimator stability experiments"
    )
    parser.add_argument("--output", default="results/",
                        help="output root directory (default: results/)")
    parser.add_argument("--repeats", type=int, default=config.SIMULATION_REPEATS,
                        help="Monte Carlo trials per injection group (task default: 100)")
    parser.add_argument("--stability-repeats", type=int,
                        default=config.SIMULATION_REPEATS,
                        help="Monte Carlo repeats per estimator stability cell")
    args = parser.parse_args()
    if args.repeats < 2 or args.stability_repeats < 2:
        parser.error("both repeat counts must be at least 2")
    output = Path(args.output)
    if not output.is_absolute():
        output = config.PROJECT_ROOT / output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_simulation(output, repeats=args.repeats,
                   stability_repeats=args.stability_repeats)


if __name__ == "__main__":
    main()
