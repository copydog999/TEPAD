"""``src.io``: dataset loaders.

Each loader's ``load()`` returns ``(df, spec)``:

* ``df``   —— cleaned pointwise ``pandas.DataFrame``;
* ``spec`` —— column-mapping arguments passed to :func:`src.preprocess.daily_split.build_curves`.
"""
from __future__ import annotations

import importlib
from typing import Callable, Dict, Tuple

import pandas as pd

#: Dataset name → submodule name (lazily imported so that ``python -m`` does not warn about duplicate imports)
LOADER_MODULES: Dict[str, str] = {
    "ding": "load_ding",
    "ds226": "load_ds226",
    "lareunion": "load_lareunion",
}


def get_loader(name: str) -> Callable[[], Tuple[pd.DataFrame, dict]]:
    """Return the ``load`` function of the given dataset.

    Parameters
    ----------
    name : str
        ``ding`` / ``ds226`` / ``lareunion``.

    Raises
    ------
    KeyError
        Unknown dataset name.
    """
    if name not in LOADER_MODULES:
        raise KeyError("unknown dataset: {!r} (known: {})".format(name, sorted(LOADER_MODULES)))
    mod = importlib.import_module("{}.{}".format(__name__, LOADER_MODULES[name]))
    return mod.load


__all__ = ["LOADER_MODULES", "get_loader"]
