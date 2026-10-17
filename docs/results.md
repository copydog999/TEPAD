# Results

Every number on this page was produced by running the code in this repository. The same values are
written to `results/logs/metrics.json` and `results/logs/baselines.json`, and
`tests/test_paper_results.py` fails if they drift.

## The three datasets

Command: `python src/pipeline/run_all.py --datasets all --output results/`

The reference statistics are fitted on all normal days of a dataset, then every day of that dataset is
scored. This is the setting of the paper's main table.

| Dataset | Days | Normal / faulty | AUC | Precision | Recall | F1 | False-alarm rate | Share faulty |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ding | 1,331 | 202 / 1,129 | 0.908 | 0.985 | 0.997 | 0.991 | 0.084 | 0.859 |
| DS226 | 44 | 12 / 32 | 1.000 | 0.889 | 1.000 | 0.941 | 0.333 | 0.818 |
| La Réunion | 247 | 224 / 23 | 0.536 | 0.548 | 1.000 | 0.708 | 0.085 | 0.170 |

The AUC scores the first check on its own. Precision, recall, F1 and the false-alarm rate describe the
combined decision. The two are reported side by side because they answer different questions: how well the
irradiance-power model separates the faults, and how well the final rule does. On La Réunion the first
check is near chance (0.536) while recall of the combined rule is 1.000, which is the clearest evidence
that the second check carries the shading case.

## Ding, check by check

| Decision | AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| First check only | 0.908 | 0.988 | 0.787 | 0.876 |
| First or second check | not ranked | 0.985 | 0.997 | 0.991 |

The combined rule raises recall from 0.787 to 0.997 at the cost of a small drop in precision. Thresholds
are 10.344 for the first check, and −0.0740, −0.0510, 0.3539, 0.9721 for the four rules of the second.

### Per-label recall

| Label | Days | First check | Second check | Combined |
| --- | ---: | ---: | ---: | ---: |
| normal | 202 | 0.054 | 0.030 | 0.084 |
| spike | 222 | 0.995 | 1.000 | 1.000 |
| inverter fault | 222 | 1.000 | 1.000 | 1.000 |
| snowy | 22 | 1.000 | 1.000 | 1.000 |
| cloudy | 219 | 0.795 | 1.000 | 1.000 |
| low value | 222 | 0.833 | 0.986 | 0.995 |
| shading | 222 | 0.288 | 0.991 | 0.991 |

The shading row is the reason the second check exists. Shading is detected in 28.8 percent of days by the
first check and 99.1 percent by the second, while the false-alarm rate on normal days moves only from
0.054 to 0.084.

### How well each residual feature separates

| Dataset | Deepest drop | Steepest fall | Negative share | Longest run | Correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ding | 0.006 | 0.004 | 0.985 | 0.981 | 0.068 |
| DS226 | 0.891 | 0.724 | 0.500 | 0.500 | 0.474 |
| La Réunion | 0.011 | 0.104 | 0.997 | 0.996 | 0.006 |

Each cell is the AUC of a single feature used on its own, so 0.5 means no separation and values near 0 or 1
mean strong separation in the direction given in [the model page](model.md). On Ding and La Réunion the
deepest drop separates in the reverse direction, which is expected: normal days have a small positive
offset and faulty days a deep negative one.

### What each check contributes

| Dataset | First check F1 | Combined F1 | Change |
| --- | ---: | ---: | ---: |
| Ding | 0.876 | 0.991 | +0.115 |
| DS226 | 0.985 | 0.941 | −0.043 |
| La Réunion | 0.158 | 0.708 | +0.550 |

The second check earns its place where faults leave the irradiance-power relation intact, and costs a
little where they do not: on DS226 the faults already break the relation, so the extra rules only add
false alarms.

## Comparison with other methods

Command: `python src/pipeline/run_baselines.py --output results/`

Fitting uses the 115 normal days of the official training split; evaluation uses all 552 days of the test
split, 87 normal and 465 faulty. Every method sees measured irradiance and power only.

| Method | Family | AUC | Precision | Recall | F1 | False-alarm rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TEPAD | this work | 0.988 | 0.951 | 1.000 | 0.975 | 0.276 |
| DBSCAN | clustering | 0.884 | 0.985 | 0.576 | 0.727 | 0.046 |
| Isolation Forest | tree ensemble | 0.849 | 0.982 | 0.475 | 0.641 | 0.046 |
| One-class SVM | kernel | 0.854 | 0.916 | 0.845 | 0.879 | 0.414 |
| LGBM, supervised | gradient boosting | 0.990 | 0.908 | 1.000 | 0.952 | 0.540 |
| Autoencoder | reconstruction | 0.736 | 0.884 | 0.899 | 0.891 | 0.632 |
| AE-LSTM-GA | reconstruction | 0.621 | 0.904 | 0.445 | 0.597 | 0.253 |
| GRU-Bayesian | reconstruction | 0.636 | 0.923 | 0.335 | 0.492 | 0.149 |
| AnoGAN | generative | 0.767 | 0.948 | 0.471 | 0.629 | 0.138 |
| Anomaly Transformer | reconstruction | 0.704 | 0.986 | 0.460 | 0.628 | 0.034 |
| Deep-SVDD | deep one-class | 0.819 | 0.847 | 1.000 | 0.917 | 0.966 |

The combined decision reaches the best F1 of the unsupervised methods and the highest recall among the
methods with a controlled false-alarm rate. LGBM is the only method trained on labels; it gains 0.002 AUC
over TEPAD but raises the false-alarm rate to 0.540.

The AUC here (0.988) differs from the 0.908 of the first table because the fitting sets differ: all 202
normal days there, only the 115 training days here. The second table is the fair comparison with the
baselines, and its numbers should not be compared with the first table's.

Because the combined decision returns yes or no, its AUC is computed from a continuous margin built from
the two checks. Precision, recall, F1 and the false-alarm rate use the decisions themselves.

## Simulations

Reports, tables and figures are written to `results/simulation/`.

Injected faults on 720-sample days, 100 repetitions per cell:

| Fault | Severity | Detected | Median delay |
| --- | --- | ---: | ---: |
| Single-sample spike | 2 σ | 0.210 | 0 |
| Single-sample spike | 3.5 σ | 0.680 | 0 |
| Single-sample spike | 5 σ | 0.970 | 0 |
| Shading | 0.70, 0.50, 0.30 of power | 1.000 | 0 |

A spike near the noise level cannot be separated from normal fluctuation, which is why the weakest level
is mostly missed. Shading is caught in every repetition.

The shading definitions experiment measures the case where the pyranometer is shaded as well; its
per-cell detection rates and delays are in `results/simulation/definitions_timeliness_report.md`. The
stability sweeps report bias, variance and mean squared error for window length, measurement noise, one
contaminated sample and sample count, in `results/simulation/tables/simulation_stability_*.csv`.

The attribution experiment uses 2,400-sample curves. Hardware, weather and shading faults shift the five
numbers in different directions, and held-out curves are assigned to the nearest of the three groups with
recall 1.000 for each. The confusion matrix and the shifts are in
`results/simulation/three_class_parameter_modes_report.md`.
