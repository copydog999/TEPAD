# How the code is organised

The repository separates the method from the scripts that run it. The estimator, the two checks and the
simulations live in their own packages and can be imported on their own; the scripts in `src/pipeline`
only call them in sequence and write files. Nothing in `src/model`, `src/detection` or `src/simulation`
imports a command-line script.

## Packages

| Package | What it contains | Main functions |
| --- | --- | --- |
| `src/config.py` | Paths, settings and the random seed, in one place | `set_seed`, `ensure_dirs` |
| `src/io/` | One loader per dataset, mapping raw columns to the common names | `get_loader(name)` |
| `src/preprocess/` | Daytime filtering, scaling, splitting a file into days | `build_curves`, `DailyCurve` |
| `src/model/` | The five-number model, the power estimate, residual features, the online distance | `estimate_5params`, `resolve_reference`, `residual_features`, `MahalanobisReference` |
| `src/detection/` | The first check, the second check, and the rule that combines them | `mahalanobis_distance`, `stage2_flags`, `run_two_level` |
| `src/evaluation/` | Metrics, per-class recall, dataset and simulation drivers | `run_dataset`, `binary_metrics` |
| `src/baselines/` | The ten comparison methods | `run_baselines` |
| `src/simulation/` | Synthetic plant, fault injection, stability sweeps, delay measurement | `simulate_normal_day`, `run_simulation` |
| `src/visualization/` | The six tables and seven figures of the paper | `make_all`, `make_cached_figures` |
| `src/pipeline/` | Command-line entry points | see below |

`src/model/mahalanobis.py` holds a version of the distance that is fitted once and then queried one
sample at a time. The live application and the Simulink model use it; the batch version, which scores a
whole dataset at once, is in `src/detection/stage1.py`. Both use the same standardisation and the same
regularisation.

## What happens to one dataset

```
src/io/load_*.py              read the raw CSVs
        │
src/preprocess/daily_split.py daytime filtering, scaling, split into days
        │
src/detection/stage1.py       ten windows × five numbers → distance → threshold at the 95th percentile
        │
src/detection/stage2.py       residual features → four threshold rules
        │
src/detection/cascade.py      faulty if either check fires
        │
src/evaluation/               metrics, per-class recall, comparison across datasets
        │
src/visualization/            tables as .tex and .csv, figures as .pdf and .png
```

The comparison with other methods runs the same detector through
`run_two_level(curves, has_label, fit_mask)`. The `fit_mask` argument restricts fitting and threshold
estimation to the training days, which is what makes the held-out protocol possible. With
`has_label=False` no labels are read at all and only the anomaly rate is reported.

## Command-line entry points

| Script | What it does |
| --- | --- |
| `src/pipeline/run_all.py` | All three datasets, optionally followed by the comparison with other methods |
| `src/pipeline/run_ding.py` | Ding only |
| `src/pipeline/run_cross.py` | DS226 and La Réunion only |
| `src/pipeline/run_baselines.py` | The comparison with other methods on Ding |
| `src/pipeline/run_simulation.py` | Injected spikes and shading, then the stability sweeps |
| `src/pipeline/run_definitions.py` | The two shading definitions and the detection delay |
| `src/pipeline/run_parameter_modes.py` | Attribution of a faulty day to hardware, weather or shading |
| `src/pipeline/publish_paper_assets.py` | Assembles the figures and tables of the paper |
| `src/pipeline/redraw_figures.py` | Redraws figures from saved metrics without rerunning anything |

`scripts/reproduce.py` calls the first six of these in order and then compares the results with the paper.

## The comparison with other methods

The comparison needs a protocol that the baselines can share, so it does not use the same fitting set as
the main experiment:

| Item | Setting |
| --- | --- |
| Fitting set | 115 normal days of the official Ding training split |
| Evaluation set | All 552 days of the official test split: 87 normal, 465 faulty |
| Input to the classical baselines | 28 summary features per day |
| Input to the deep baselines | Two channels, irradiance and power, resampled to 96 points |
| Threshold | 95th percentile of the baseline's own scores on the fitting set |
| Labels | Used only by LGBM |

Days in the two splits do not overlap, and the pairs of site and date in the two splits are disjoint, so
no day is seen during fitting and evaluation. All baselines see measured irradiance and power only, not
the modelled power channel; the detector uses the modelled channel through its second check, which is
what the paper reports as the complete method.

The detector's own decision is a yes or no per day, which cannot be ranked and therefore has no natural
AUC. To report one, the code builds a continuous margin from the two checks and ranks by it; precision,
recall, F1 and the false-alarm rate use the real decisions.

## The simulations

Three scripts, none of which reads a label.

`run_simulation.py` works on synthetic days of 720 one-minute samples, with the healthy parameters
$(\alpha,\beta,\eta)=(8,2,0.82)$. It injects single-sample spikes at 2, 3.5 and 5 standard deviations,
shading of 10 or 60 samples reducing power to 0.7, 0.5 or 0.3 of its value, and combinations of the two,
rotating the injection time between 20, 40 and 60 percent of the day, with 100 repetitions per cell. It
reports the detection rate, the delay and the false-alarm rate on fault-free days. It then sweeps the
estimator: window length 10 to 100, measurement noise 0.01 to 0.2, one contaminated sample at 1, 3 or 5
standard deviations, and sample counts from 10 to 1,000, reporting bias, variance and mean squared error
for each of the five numbers.

`run_definitions.py` separates two cases that shading can mean. In the first, only the modules are
shaded: power falls while the irradiance reading stays correct, and the irradiance-power relation breaks.
In the second, the pyranometer is shaded too: irradiance and power fall together, the relation survives,
and only the comparison with an independent power estimate can see the fault. Each case is run 100 times
per combination of intensity, duration and time of day. The script reports the mean time to detection and
the share of events detected within 1, 5, 10, 20 and 50 samples.

`run_parameter_modes.py` generates 2,400-sample curves with hardware, weather or shading faults, learns
the typical five-number shift of each kind from a training set, and assigns held-out curves to the
nearest of the three. It reports the shift directions, the confusion matrix and per-class recall, then
applies the same rule to the three datasets.

## Reproducibility

`src/config.py` is the only place where a path or a setting is written down, and `set_seed(42)` fixes
Python, NumPy and PyTorch. The simulations use their own generators with explicit seeds, so two runs with
the same seed produce identical files. The deep baselines cache their scores in `results/models/`; the
cache key includes the hyperparameters and a hash of the data, so a changed setting or a changed dataset
forces a retrain instead of silently reusing an old result.
