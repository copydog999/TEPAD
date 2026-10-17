# -*- coding: utf-8 -*-
"""Fetch the raw data of the three datasets into ``datasets/<name>/raw/``.

The datasets are public but too large to keep in git: about 196 MB for Ding,
74 MB for DS226 and 2.5 GB for La Reunion. This script downloads them from
Zenodo, renames the files to the names the loaders expect, and checks every
file against the MD5 published in the Zenodo record.

Usage
-----
    python scripts/download_datasets.py --download                 # all three
    python scripts/download_datasets.py --download --datasets ds226 lareunion
    python scripts/download_datasets.py --list                     # sizes and URLs only
    python scripts/download_datasets.py --verify                   # check local files

A local copy can also be laid out from disk instead of downloaded:

    python scripts/download_datasets.py --source /path/to/data --mode link
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

#: Zenodo record for each dataset, as cited in the paper.
RECORDS: Dict[str, str] = {
    "ding": "10.5281/zenodo.8248701",
    "ds226": "10.5281/zenodo.19259504",
    "lareunion": "10.5281/zenodo.7828879",
}

#: Published file name on Zenodo -> path relative to ``datasets/<name>/raw/``.
LAYOUT: Dict[str, Dict[str, str]] = {
    "ding": {
        "enhance_12times_{label}_anomaly_data_of_{sites}sites.csv":
            "{split}/{label}_{split}.csv",
        "enhance_12times_smooth_intense_normal_data_of_{sites}sites_clean.csv":
            "{split}/normal_{split}.csv",
    },
    "ds226": {
        "df_ia_raw_acquisition.csv": "df_ia_raw_acquisition.csv",
        "PV_Diagnostic_Final_Labels.csv": "PV_Diagnostic_Final_Labels.csv",
    },
    "lareunion": {
        "dt1_solar_and_meteorological_measurement.csv": "dt1_solar_met.csv",
        "dt2_electrical_production_inverter_1_with_faults.csv": "dt2_inverter1_faults.csv",
        "dt3_electrical_production_inverter_2.csv": "dt3_inverter2_normal.csv",
    },
}

#: Our class name -> the spelling used in the published file names.
DING_CLASSES = {
    "spike": "spike",
    "inverter_fault": "inverter_fault",
    "snowy": "snowy",
    "cloudy": "cloudy",
    "low_value": "lowValue",
    "shading": "shading",
}
DING_SPLITS = {"train": "20train", "test": "10test"}


def _record(doi: str) -> dict:
    """Read the metadata of a Zenodo record.

    The API is addressed by the numeric record id, which is the part after the
    last dot of the DOI (``10.5281/zenodo.8248701`` -> ``8248701``).
    """
    record_id = doi.rsplit(".", 1)[-1]
    url = "https://zenodo.org/api/records/" + record_id
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def _published(record: dict) -> Dict[str, Tuple[str, str, int]]:
    """Map published file name -> (URL, md5, size)."""
    out: Dict[str, Tuple[str, str, int]] = {}
    for item in record.get("files", []):
        checksum = item.get("checksum", "")
        md5 = checksum.split(":", 1)[1] if ":" in checksum else ""
        out[item["key"]] = (item["links"]["self"], md5, int(item["size"]))
    return out


def _lookup(published: Dict[str, Tuple[str, str, int]], key: str,
            destination: str) -> Tuple[str, str, str, int]:
    """Return (published name, destination, md5, size), failing loudly when absent."""
    if key not in published:
        raise KeyError("{} is not in the Zenodo record".format(key))
    _url, md5, size = published[key]
    return key, destination, md5, size


def expected_files(name: str, record: dict) -> List[Tuple[str, str, str, int]]:
    """Return (published name, destination, md5, size) for one dataset."""
    published = _published(record)
    plan: List[Tuple[str, str, str, int]] = []

    if name == "ding":
        for split, sites in DING_SPLITS.items():
            for label, published_name in DING_CLASSES.items():
                key = "enhance_12times_{}_anomaly_data_of_{}sites.csv".format(
                    published_name, sites)
                plan.append(_lookup(published, key, "{}/{}_{}.csv".format(split, label, split)))
            key = "enhance_12times_smooth_intense_normal_data_of_{}sites_clean.csv".format(sites)
            plan.append(_lookup(published, key, "{}/normal_{}.csv".format(split, split)))
        return plan

    for key, template in LAYOUT[name].items():
        plan.append(_lookup(published, key, template))
    return plan


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dst: Path, size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_suffix(dst.suffix + ".part")
    print("  [get] {:.0f} MB -> {}".format(size / 1e6, dst))
    with urllib.request.urlopen(url, timeout=120) as src, partial.open("wb") as out:
        done = 0
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if size:
                print("\r    {:5.1f}%".format(100.0 * done / size), end="")
    print("\r    100.0%")
    partial.replace(dst)


def fetch(names: List[str]) -> int:
    """Download the files of ``names`` and check their checksums."""
    problems = 0
    for name in names:
        print("[{}] Zenodo {}".format(name, RECORDS[name]))
        record = _record(RECORDS[name])
        urls = _published(record)
        raw = config.dataset_raw_dir(name)
        for key, rel, md5, size in expected_files(name, record):
            dst = raw / rel
            if dst.exists():
                print("  [have] {} ({} MB)".format(rel, round(dst.stat().st_size / 1e6)))
            else:
                _download(urls[key][0], dst, size)
            if md5 and _md5(dst) != md5:
                print("  [FAIL] checksum mismatch: {}".format(rel))
                problems += 1
            else:
                print("  [ok] {}".format(rel))
    return problems


def verify(names: List[str]) -> int:
    """Check the files already on disk against the published checksums."""
    problems = 0
    for name in names:
        print("[{}]".format(name))
        record = _record(RECORDS[name])
        raw = config.dataset_raw_dir(name)
        for _key, rel, md5, _size in expected_files(name, record):
            dst = raw / rel
            if not dst.exists():
                print("  [missing] {}".format(rel))
                problems += 1
            elif _md5(dst) != md5:
                print("  [differs] {} (local copy is not the published file)".format(rel))
            else:
                print("  [ok] {}".format(rel))
    return problems


def show_plan(names: List[str]) -> None:
    """Print sizes and destinations without downloading anything."""
    total = 0
    for name in names:
        record = _record(RECORDS[name])
        print("[{}] {}".format(name, RECORDS[name]))
        for key, rel, _md5, size in expected_files(name, record):
            total += size
            print("  {:>8.1f} MB  {}  ->  {}".format(size / 1e6, key, rel))
    print("\ntotal: {:.0f} MB".format(total / 1e6))


def prepare_local(source: Path, names: List[str], mode: str) -> None:
    """Lay out each dataset from a local directory instead of downloading."""
    def place(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print("  [skip] {}".format(dst.name))
            return
        if mode == "link":
            try:
                os.link(src, dst)
                print("  [link] {}".format(dst.name))
                return
            except OSError:
                print("  [copy] {} (link not possible)".format(dst.name))
        shutil.copy2(src, dst)

    for name in names:
        raw = config.dataset_raw_dir(name)
        print("[{}] -> {}".format(name, raw))
        if name == "ding":
            for split in ("train", "test"):
                for f in sorted((source / "ding" / split).glob("*.csv")):
                    place(f, raw / split / f.name)
        elif name == "ds226":
            for f in sorted((source / "ds226").glob("*.csv")):
                place(f, raw / f.name)
        elif name == "lareunion":
            for f in sorted((source / "lareunion").glob("dt*.csv")):
                place(f, raw / f.name)
        else:
            print("  [warn] unknown dataset {}".format(name))


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare the TEPAD datasets")
    ap.add_argument("--datasets", nargs="*", default=list(RECORDS), choices=list(RECORDS),
                    help="which datasets to prepare (default: all)")
    ap.add_argument("--download", action="store_true", help="download from Zenodo")
    ap.add_argument("--list", action="store_true", help="show sizes and destinations only")
    ap.add_argument("--verify", action="store_true", help="check local files against Zenodo")
    ap.add_argument("--source", default="", help="local directory to copy from instead")
    ap.add_argument("--mode", choices=["link", "copy"], default="link")
    args = ap.parse_args()

    config.ensure_dirs()
    if args.list:
        show_plan(args.datasets)
        return 0
    if args.verify:
        return 1 if verify(args.datasets) else 0
    if args.source:
        prepare_local(Path(args.source), args.datasets, args.mode)
        return 0
    if args.download:
        return 1 if fetch(args.datasets) else 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
