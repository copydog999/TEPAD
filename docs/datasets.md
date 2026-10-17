# Datasets

Three public datasets are used in the paper, referred to there as A, B and C.

| Dataset | In the paper | Zenodo | Used for | Days | Fault labels |
| --- | --- | --- | --- | ---: | --- |
| Ding | Dataset A | [10.5281/zenodo.8248701](https://doi.org/10.5281/zenodo.8248701) | Main experiment, comparison with other methods | 1,331 | 7 classes |
| DS226 | Dataset B | [10.5281/zenodo.19259504](https://doi.org/10.5281/zenodo.19259504) | Cross-dataset check | 44 | 5 classes |
| La Réunion | Dataset C | [10.5281/zenodo.7828879](https://doi.org/10.5281/zenodo.7828879) | Cross-dataset check | 247 | 7 shading codes |

Ding has 202 normal days and 1,129 faulty ones. DS226 has 12 normal and 32 faulty. La Réunion has 224
normal and 23 faulty, all of them shading.

## The data is downloaded, not committed

The raw files total about 2.9 GB, so they are not part of the repository. One command fetches all three,
renames the published files to the names the loaders use, and checks each file against the MD5 published
in the Zenodo record:

```bash
python scripts/download_datasets.py --download
```

To see what would be downloaded, without fetching anything:

```bash
python scripts/download_datasets.py --list
```

To check files that are already on disk:

```bash
python scripts/download_datasets.py --verify
```

If the data already exists elsewhere, lay it out from there instead. `link` creates hard links, so no
extra disk space is used:

```bash
python scripts/download_datasets.py --source /path/to/data --mode link
```

A directory passed this way must contain `ding/train/*.csv`, `ding/test/*.csv`, `ds226/*.csv` and
`lareunion/dt*.csv`.

### How the published files map onto the loaders

| Dataset | Published file | Used as |
| --- | --- | --- |
| Ding | `enhance_12times_<class>_anomaly_data_of_20trainsites.csv` | `train/<class>_train.csv` |
| Ding | `enhance_12times_<class>_anomaly_data_of_10testsites.csv` | `test/<class>_test.csv` |
| Ding | `enhance_12times_smooth_intense_normal_data_of_20trainsites_clean.csv` | `train/normal_train.csv` |
| Ding | `enhance_12times_smooth_intense_normal_data_of_10testsites_clean.csv` | `test/normal_test.csv` |
| DS226 | `df_ia_raw_acquisition.csv` | same name |
| DS226 | `PV_Diagnostic_Final_Labels.csv` | same name |
| La Réunion | `dt1_solar_and_meteorological_measurement.csv` | `dt1_solar_met.csv` |
| La Réunion | `dt2_electrical_production_inverter_1_with_faults.csv` | `dt2_inverter1_faults.csv` |
| La Réunion | `dt3_electrical_production_inverter_2.csv` | `dt3_inverter2_normal.csv` |

The Ding classes are `spike`, `inverter_fault`, `snowy`, `cloudy`, `low_value` and `shading`; the
published files spell `low_value` as `lowValue`.

## The La Réunion analysis window

The irradiance series on Zenodo runs until 2023-04-14, which is longer than the inverter series that
provides the reference. The reported results were obtained on the overlap that ends where the irradiance
export available at the time ended, giving 247 days. That window is fixed in `src/config.py` as
`LAREUNION_TIME_WINDOW`, so downloading the longer published file still produces the same 247 days.
Without the fixed window the extra months would change the day count and every number in the La Réunion
column.

## How a file becomes days

The same four steps are applied to all three datasets, in `src/preprocess/`:

1. Keep only daytime samples, defined as irradiance above its own 5th percentile.
2. Scale irradiance and power to $[0,1]$ with a single min-max per dataset, so modelled and measured
   power land on the same scale.
3. Group samples by site, date and, where present, label.
4. Drop days with fewer than 10 samples.

For Ding only, groups longer than 864 samples are dropped and rows that are exact duplicates on the
feature columns are removed.

Each dataset supplies a different independent power estimate for the second check:

| Dataset | Estimate | Source |
| --- | --- | --- |
| Ding | Modelled power shipped with the data | `modeled_power` column |
| DS226 | None published; falls back to the fitted relation | $\min(\eta g, 1)$ |
| La Réunion | Healthy twin inverter | `Pg` of inverter 2 |

## Labels used for the attribution analysis

Fault labels are never used for fitting or for choosing thresholds. One optional analysis, reported in the
paper as the parameter distribution, groups the raw labels by the part of the plant they affect:

| Dataset | Hardware | Weather | Shading |
| --- | --- | --- | --- |
| Ding | `spike`, `inverter_fault`, `low_value` | `cloudy`, `snowy` | `shading` |
| DS226 | `Degradation`, `Open Circuit`, `Short Circuit` | none | `Shading` |
| La Réunion | none | none | all seven codes: 1.0, 2.1, 2.2, 2.3, 3.1, 3.2, 4.0 |

The grouping is defined in `src/evaluation/parameter_modes.py`. Labels are read only after predictions have
been made, to score them.

## What to expect from each dataset

Ding is the only dataset whose faults cover several physical mechanisms, and the only one shipped with a
modelled power channel. That is why it carries the main experiment.

DS226 contains faults that change the irradiance-power relation directly, so the first check separates
them almost perfectly. It is also small: 44 days, 12 of them normal. Treat its numbers as indicative
rather than precise.

La Réunion contains shading only, which leaves the irradiance-power relation intact. The first check is
close to chance there (AUC 0.536) and almost all of the detection comes from the second check. Recall
reaches 1.000, but with 23 faulty days out of 247 the precision is only 0.548: the false-alarm rate is
0.085, similar to Ding, but there are far fewer faults to balance it.
