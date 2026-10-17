#!/usr/bin/env bash
# TEPAD one-click run (Linux / macOS)
set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS="${1:-all}"
EXTRA="${2:-}"
echo "=== TEPAD pipeline (datasets=${DATASETS} ${EXTRA}) ==="

if [ ! -d "datasets/ding/raw/train" ]; then
  echo "[!] datasets are not ready; run first: python scripts/download_datasets.py --source <dir>"
fi

python src/pipeline/run_all.py --datasets "${DATASETS}" --output results/ ${EXTRA}
echo "[ok] results written to results/"
# Note: passing --baselines as the 2nd argument also runs the Ding cross-model comparison (table 6 / figure 7, about 6-8 minutes)
