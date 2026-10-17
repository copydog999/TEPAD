# Simulink model

A third implementation of the detector, next to the Python code and the C++ demo: a discrete Simulink
model with a synthetic PV plant, manual fault injection, and the two-window detector. One step is one
minute and a day is 720 steps. The paper's experiments do not use it.

The model is built by script. `src/tepadsim_build_model.m` contains the block topology, so the `.slx`
file is an output rather than a source of truth; reviewing the model means reading that script. The two
MATLAB Function blocks read their code from `tepadsim_plant_step.m` and `tepadsim_detector_step.m`, so the
algorithm inside the model is the same file that the command-line test runs.

## Running it

Double-click `simulink\run_simulink.bat`, or run the PowerShell script directly:

```powershell
powershell -ExecutionPolicy Bypass -File simulink\run_simulink.ps1
```

It starts MATLAB in batch mode and returns the exit code, which is zero when everything passed.

| Step | What runs | Output |
| --- | --- | --- |
| 1 | `tepadsim_selftest` compares the MATLAB detector against saved Python output | Four PASS lines |
| 2 | `tepadsim_build_model` builds the model | `model/tepadsim_pv_tepad.slx` |
| 3 | `tepadsim_verify_model` checks the built model against the pure-MATLAB path | `data/model_checks.csv` |
| 4 | `tepadsim_injection_scan` runs spikes, shading and a mixed case | `data/injection_scan.csv` |
| 5 | `tepadsim_plot_scan` draws the two figures | `figures/*.png`, `figures/*.pdf` |

Useful switches: `-Quick` stops after step 3, `-NoRebuild` reuses an existing `.slx`,
`-Target <name>` runs a single step, `-MatlabRoot <path>` points at a specific MATLAB installation.

## What the model contains

A clear-sky irradiance profile modulated by a cloud process, a PVWatts-style conversion with a
temperature correction and a noise term whose spread grows with output, and the detector from
`src/simulation/model.py`. Faults are injected as a spike at a chosen step or as an attenuation of power
over a number of steps, with irradiance left unchanged.

The detector keeps two causal windows of 10 and 60 steps, estimates the five numbers in each, and combines
a Mahalanobis distance on the window features with a residual score for the current sample. The alarm
fires when any of the three channels crosses its threshold.

## Files

| Path | Contents |
| --- | --- |
| `run_simulink.ps1`, `run_simulink.bat` | One-command entry point; logs go to `logs/` |
| `src/tepadsim_config.m` | Paths, constants and injection defaults, aligned with `src/config.py` |
| `src/tepadsim_plant_step.m` | One step of the plant; also the body of a MATLAB Function block |
| `src/tepadsim_detector_step.m` | One step of the detector; likewise |
| `src/tepadsim_build_model.m` | Builds the model programmatically |
| `src/tepadsim_simulate.m` | Runs the model from the command line |
| `src/tepadsim_selftest.m`, `tepadsim_verify_model.m` | The checks done in steps 1 and 3 |
| `src/tepadsim_injection_scan.m`, `tepadsim_plot_scan.m` | The scan in step 4 and the figures in step 5 |
| `src/tepadsim_load_calibration.m`, `tepadsim_load_golden.m` | Read the shared files under `sim_ui/`; there is no second copy of the reference data |
| `model/`, `data/`, `figures/`, `logs/` | Outputs; `model/*.slx` and `logs/` are not tracked |

The calibration and reference files are shared with the C++ demo through `sim_ui/calibration/` and
`sim_ui/selftest/golden/`, so the two implementations are checked against the same numbers.
