# -*- coding: utf-8 -*-
"""Entry point for ``python -m src.io``: list the available datasets and try loading one."""
from __future__ import annotations

import sys

from . import LOADER_MODULES, get_loader


def main(argv=None) -> int:
    """Print the available datasets; the optional argument is the dataset to try loading."""
    argv = list(argv if argv is not None else sys.argv[1:])
    print("available datasets:", sorted(LOADER_MODULES))
    if argv:
        df, spec = get_loader(argv[0])()
        print("loaded {!r}: {} rows".format(argv[0], len(df)))
        print("spec:", {k: v for k, v in spec.items() if k != "class_names"})
    return 0


if __name__ == "__main__":  # pragma: no cover - minimal example
    raise SystemExit(main())
