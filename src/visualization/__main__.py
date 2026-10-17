# -*- coding: utf-8 -*-
"""Entry point for ``python -m src.visualization``: lists the figure/table scripts."""
from __future__ import annotations

from . import FIGURE_MODULES, TABLE_MODULES


def main() -> int:
    """Print the names of all table and figure generation scripts."""
    print("tables :", TABLE_MODULES)
    print("figures:", FIGURE_MODULES)
    return 0


if __name__ == "__main__":  # pragma: no cover - minimal example
    raise SystemExit(main())
