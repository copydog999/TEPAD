"""Re-render experiment figures from existing result artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.visualization import make_cached_figures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redraw all available TEPAD figures from saved experiment outputs"
    )
    parser.add_argument("--output", default="results/",
                        help="results directory (default: results/)")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = config.PROJECT_ROOT / output
    created = make_cached_figures(output)
    if not created:
        raise RuntimeError("No figures were generated; check that result JSON files exist")
    print("Redrew {} figure files".format(len(created)))


if __name__ == "__main__":
    main()
