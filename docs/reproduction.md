# Reproducing the results

This page lists what to install, what data to fetch, the commands to run, and the numbers to expect. The
whole sequence is five commands.

## Requirements

Python 3.8 or newer. The code was developed and checked on 3.8.10.

```bash
python -m pip install -r requirements.txt
```

This installs NumPy, pandas, SciPy, scikit-learn, matplotlib, LightGBM and PyTorch. The figures use Times
New Roman; if the font is missing matplotlib prints a warning and substitutes another serif font, which
changes the look but not the numbers.

## Data

All three datasets download from Zenodo in one command. The script renames the published files to the
names the loaders expect and checks each file against the MD5 in the Zenodo record. The total is about
2.9 GB, most of it the La Réunion irradiance series.

```bash
python scripts/download_datasets.py --download
```

Useful variants:

```bash
python scripts/download_datasets.py --list                  # sizes and destinations, no download
python scripts/download_datasets.py --verify                # check what is already on disk
python scripts/download_datasets.py --download --datasets ds226    # one dataset only
python scripts/download_datasets.py --source /path/to/data --mode link   # from a local copy
```

The DOIs, the published file names and the mapping onto the loaders are listed in
[datasets.md](datasets.md), together with the fixed analysis window used for La Réunion.

Every step reads the data from `datasets/`. Nothing is downloaded at run time.

## Running everything

```bash
python scripts/reproduce.py
```

The script runs the five steps below in order and stops at the first failure.

| Step | Script | Produces |
| --- | --- | --- |
| 1 | `run_all.py --datasets all --baselines` | Detection on the three datasets, then the baseline comparison |
| 2 | `run_simulation.py` | Injected faults and the stability sweeps |
| 3 | `run_definitions.py` | The two shading definitions and the detection delay |
| 4 | `run_parameter_modes.py` | Attribution to hardware, weather or shading |
| 5 | `publish_paper_assets.py` | The figures and tables of the paper |

With 100 repetitions per cell the simulations take tens of minutes; the three datasets take about two
minutes, and the deep baselines take several minutes the first time and seconds afterwards because their
scores are cached in `results/models/`.

`--quick` reduces the repetitions to five and finishes in a few minutes. Use it to check that an
installation works; do not report its numbers.

## The numbers to expect

After the last step the script prints a comparison against these values, with a tolerance of 0.002:

| Dataset | AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Ding | 0.908 | 0.985 | 0.997 | 0.991 |
| DS226 | 1.000 | 0.889 | 1.000 | 0.941 |
| La Réunion | 0.536 | 0.548 | 1.000 | 0.708 |

If any value is outside tolerance the script exits with a non-zero status. The measured values are always
written to `results/logs/metrics.json`, so they can be checked without relying on the script.

Pass `--skip-checks` to print the comparison without failing on a difference, which is useful while
debugging a modified copy.

## Running one part at a time

Every entry point accepts `--output <dir>` and can be run on its own.

```bash
python src/pipeline/run_all.py            --datasets all --output results/
python src/pipeline/run_all.py            --datasets ding --output results/
python src/pipeline/run_all.py            --datasets cross --output results/
python src/pipeline/run_baselines.py      --output results/
python src/pipeline/run_simulation.py     --output results/
python src/pipeline/run_definitions.py    --output results/
python src/pipeline/run_parameter_modes.py --output results/
python src/pipeline/redraw_figures.py     --output results/
```

`--datasets cross` means DS226 and La Réunion; `--datasets all` means all three.

### The baseline cache

The six deep baselines train on the fitting days and store their weights and scores in `results/models/`.
A cache file is keyed by the model name, its hyperparameters and a hash of the data, so changing either
forces a retrain. To ignore the cache and retrain:

```bash
python src/pipeline/run_baselines.py --output results/ --no-cache
```

Running only a subset of the baselines overwrites `baselines.json` and therefore the comparison table:

```bash
python src/pipeline/run_baselines.py --output results/ --models "Isolation Forest,Autoencoder"
```

## Common problems

| Symptom | Cause and fix |
| --- | --- |
| `Required raw data is missing` | The script lists the missing files; run `download_datasets.py --download` |
| `HTTP Error 404` while downloading | A dataset DOI moved; check `--list` and the links in [datasets.md](datasets.md) |
| `checksum mismatch` after downloading | The download was interrupted; delete the file and rerun |
| `TEPAD requires Python 3.8 or newer` | The active interpreter is too old |
| `ModuleNotFoundError: lightgbm` or `torch` | Dependencies were not installed, or the wrong environment is active |
| Font warnings while plotting | Times New Roman is not installed; the figures still render |
| `LinAlgError: SVD did not converge` | A dataset ended up with no normal days; check that label 0 means normal |
| A run takes a long time | La Réunion alone needs about a minute of resampling; the simulations dominate the rest |
| Numbers unchanged after editing a setting | A cached baseline score is being reused; add `--no-cache` |
| The comparison fails but the values look close | The tolerance is 0.002; check that all fourteen Ding files are present and that no old cache is being used |

## What makes a run repeatable

The paths and every setting are in `src/config.py`, and `set_seed(42)` fixes Python, NumPy and PyTorch.
The simulations use their own generators with explicit seeds, so repeated runs produce identical files.
Tables are written from the measured values at run time; no number in this repository is typed in by hand.
The fitting and evaluation days of the baseline comparison are disjoint and come from the official split,
so no day is used twice.

`tests/test_paper_results.py` recomputes the three-dataset metrics and fails if any of them moves by more
than 0.001. Run the tests with:

```bash
python -m unittest discover -s tests
```
