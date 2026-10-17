# TEPAD: A Physics-Interpretable, Dual-Scale Unsupervised Power Anomaly Detector for Photovoltaic Systems

TEPAD finds faults in photovoltaic (PV) power data. It is fitted on normal days only: no fault labels
are used and no neural network is trained. It learns how irradiance normally turns into power, and then
reports a day when that relationship no longer holds, or when the measured power falls below what an
independent power estimate says it should be.

This repository contains the code behind the paper: the experiments on three public datasets, the
comparison against ten baseline methods, and the fault-injection simulations.

Documentation: [model](docs/model.md) · [datasets](docs/datasets.md) · [code layout](docs/pipeline.md)
· [results](docs/results.md) · [reproduction](docs/reproduction.md)

## How it works

**Five numbers per window.** A day is cut into ten overlapping windows. In each window we estimate five
numbers that describe the local behaviour of the plant: two shape parameters of the irradiance
distribution within the window, the slope of power against irradiance (the conversion efficiency), and
two parameters describing how much power fluctuates. A full day is therefore described by 50 numbers.

**First check.** On normal days these 50 numbers scatter around a mean vector with correlations between
them. Each new day is scored by its Mahalanobis distance from that normal cloud, which is the usual
multivariate distance that accounts for those correlations. The threshold is the 95th percentile of the
distance measured on normal days.

**Second check.** The first check only sees the relation between irradiance and power. If a pyranometer
and the modules are shaded at the same time, irradiance and power fall together, the relation still looks
normal, and the fault is invisible. To cover this case the code compares measured power with an external
power estimate for the same weather, and describes the difference by its depth, its steepest fall, the
fraction of samples that are negative, the longest negative run, and the correlation with the estimate.
Thresholds are the 1st and 99th percentiles of normal days.

A day is reported as faulty if either check fires. Both checks use the same five numbers, so the method
stays a single model with two decision rules rather than two independent detectors.

## Install and run

Python 3.8 or newer is required.

```bash
python -m pip install -r requirements.txt
python scripts/download_datasets.py --download
python scripts/reproduce.py
```

The first command installs the dependencies, the second fetches all three datasets from Zenodo (about
2.9 GB, checksummed), and the third runs every experiment. If the data is already on disk, lay it out
from there instead of downloading:

```bash
python scripts/download_datasets.py --source /path/to/data --mode link
```

[datasets.md](docs/datasets.md) lists the DOIs, the file names and the mapping onto the loaders.

`scripts/reproduce.py` runs the experiments in order:

| Step | What it runs |
| --- | --- |
| 1 | Detection on the three datasets, then the comparison against the baselines |
| 2 | Injected spikes and shading, plus the estimator stability sweeps |
| 3 | Two shading definitions and detection delay |
| 4 | Attribution of a detected day to hardware, weather or shading |
| 5 | Figures and tables for the paper |

Expect roughly an hour on a laptop at the default 100 repetitions. Add `--quick` to use five repetitions
and finish in a few minutes; the numbers from a quick run are not meant to be reported.

## Results

### The three datasets

The reference statistics (mean vector, correlations and thresholds) are fitted on all normal days of each
dataset, then every day of that dataset is scored.

| Dataset | Days | Normal / faulty | AUC | Precision | Recall | F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Ding | 1,331 | 202 / 1,129 | 0.908 | 0.985 | 0.997 | 0.991 |
| DS226 | 44 | 12 / 32 | 1.000 | 0.889 | 1.000 | 0.941 |
| La Réunion | 247 | 224 / 23 | 0.536 | 0.548 | 1.000 | 0.708 |

The AUC column scores the first check alone; precision, recall and F1 describe the final decision after
both checks. La Réunion contains only shading faults, which by construction leave the irradiance-power
relation intact, so the first check is close to chance there (AUC 0.536) and the second check does the
work. Its precision is low because only 23 of 247 days are faulty: the false-alarm rate is 0.085, similar
to Ding, but the anomalies are far rarer, so the same false-alarm rate costs much more precision.

### Comparison with other methods

Here the reference is fitted on the 115 normal days of the official training split and evaluated on the
552 days of the test split, because the baseline methods need a held-out protocol. All non-supervised
methods see only measured irradiance and power; LGBM additionally sees the training labels.

| Method | AUC | F1 |
| --- | ---: | ---: |
| TEPAD | 0.988 | **0.975** |
| DBSCAN | 0.884 | 0.727 |
| Isolation Forest | 0.849 | 0.641 |
| One-class SVM | 0.854 | 0.879 |
| LGBM (supervised) | **0.990** | 0.952 |
| Autoencoder | 0.736 | 0.891 |
| AE-LSTM-GA | 0.621 | 0.597 |
| GRU-Bayesian | 0.636 | 0.492 |
| AnoGAN | 0.767 | 0.629 |
| Anomaly Transformer | 0.704 | 0.628 |
| Deep-SVDD | 0.819 | 0.917 |

The numbers in the two tables differ for TEPAD (0.908 versus 0.988) because the fitting sets differ: all
202 normal days in the first table, only the 115 training days in the second. The second table is the
honest like-for-like comparison with the baselines, and the two are not directly comparable to each other.

The final decision on the test split is a yes/no answer per day, which has no natural ranking score.
To report an AUC for it, the code ranks days by a continuous margin built from the two checks; the
precision, recall and F1 columns use the actual decisions, not that margin.

## What a run produces

| Output | Path |
| --- | --- |
| Metrics for the three datasets | `results/logs/metrics.json` |
| Baseline results, including thresholds and timings | `results/logs/baselines.json` |
| Tables and figures | `results/tables/`, `results/figures/` |
| Per-day scores and decisions | `datasets/<dataset>/processed/daily_curves.csv` |
| Simulation reports, tables and figures | `results/simulation/` |
| Published paper assets | `research_paper/tables/`, `research_paper/figures/` |

After the last step the script compares the measured AUC, precision, recall and F1 of the three datasets
with the values in the first table, within a tolerance of 0.002, and prints the comparison. If anything is
outside tolerance it exits with a non-zero status. The measured values are written to
`results/logs/metrics.json` either way, so they can be checked independently.

## How the code is organised

```
src/model/          five-number estimation, external power estimate, residual features
src/detection/      the first check, the second check, and the rule that combines them
src/preprocess/     daylight filtering, normalization, splitting a file into days
src/io/             one loader per dataset
src/evaluation/     metrics, per-class recall, dataset and simulation drivers
src/baselines/      the ten comparison methods
src/simulation/     synthetic plant, fault injection, stability sweeps, delay measurement
src/visualization/  the figures and tables of the paper
src/pipeline/       command-line entry points; each one only calls the modules above
scripts/            one-command reproduction and dataset download
sim_ui/             C++ desktop application that runs the same detector live
simulink/           Simulink model of the same detector
tests/              unit tests, including a regression test on the three-dataset numbers
```

The estimator, the two checks and the simulations can be imported and used on their own; nothing in
`src/model`, `src/detection` or `src/simulation` depends on the command-line scripts.

## What is not in this repository

The raw data files, which are downloaded by the command above, and the manuscript sources. The `sim_ui` and `simulink`
directories are separate implementations of the same detector, used for demonstration and cross-checking;
the paper experiments do not call them.

## Limitations

The simulations are generated from the same parametric model that the method assumes, so they measure
how well the estimator and the decision rules behave, not how the method performs on arbitrary field data.
The three datasets are small, and the DS226 result rests on 44 days, 12 of them normal. The method also
needs a usable external power estimate; where a dataset has no measured or modelled power, the code falls
back to a physical estimate derived from the fitted irradiance-power relation, which is weaker than a
real forecast and is the main reason the La Réunion and DS226 results behave differently from Ding.

## License

See [LICENSE](LICENSE).
