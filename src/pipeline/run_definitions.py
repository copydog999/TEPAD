"""Run the E3 (simulation completion) and E5 (timeliness) experiments."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.simulation.definitions_experiment import (  # noqa: E402
    run_definition_and_timeliness,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TEPAD E3 simulated-injection completion and E5 timeliness evaluation"
    )
    parser.add_argument("--output", default="results/",
                        help="results root (default: results/)")
    parser.add_argument("--repeats", type=int, default=config.SIMULATION_REPEATS,
                        help="Monte Carlo trials per cell (task default: 100)")
    parser.add_argument("--noise-repeats", type=int, default=config.SIMULATION_REPEATS,
                        help="trials per noise-sweep cell")
    args = parser.parse_args()
    if args.repeats < 2 or args.noise_repeats < 2:
        parser.error("both repeat counts must be at least 2")
    output = Path(args.output)
    if not output.is_absolute():
        output = config.PROJECT_ROOT / output
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    run_definition_and_timeliness(output, repeats=int(args.repeats),
                                  noise_repeats=int(args.noise_repeats))


if __name__ == "__main__":
    main()
