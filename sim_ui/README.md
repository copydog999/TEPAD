# Live detection demo (Windows, C++)

A small MFC application that runs the detector on a stream instead of on a finished dataset. It receives
`tick,G,P` rows, updates two sliding windows, computes the same distance as the batch code, and plots the
curves and the alarm state as they arrive. It is a demonstration of the online behaviour; the paper's
experiments do not use it.

The detector itself is a line-by-line port of `src/simulation/model.py`: two causal windows of 10 and 60
samples, exponentially weighted parameter estimates, a Mahalanobis distance over the window features plus
a residual score for the current sample, and an alarm when any of the three channels crosses its own
threshold.

## Running it

Double-click `scripts\run_sim.bat`. It does four things in order:

1. Exports the calibration file if it is not there yet.
2. Builds `sim_ui.exe` with MSBuild if it is missing.
3. Starts a bridge on `127.0.0.1:18801` that replays one simulated day of 720 samples in a loop.
4. Opens the application with `--auto`, so it connects to the bridge on its own.

Curves start moving as soon as the window opens. The information row shows the calibrated thresholds once
the calibration file has loaded. Closing the bridge window stops the stream and the application reports
the link as disconnected.

The bridge replays a normal day and injects nothing, so any alarm you see comes from something you did.

## Injecting a fault by hand

The injection row at the bottom is there to be used:

| Control | Effect |
| --- | --- |
| Number box and **Inject spike** | Adds a multiple of the calibrated residual spread to the next sample |
| Number box and **Inject shade** | Multiplies power by the given factor for that many samples |

The text to the right of the buttons reports what happened: it counts the samples since the injection and
then states either the delay to the first alarm or that no alarm appeared before the deadline of 180
samples. One sample is one minute in the simulation.

When you inject, the view narrows to 120 samples and marks the fault on the curve, because a single
sample is invisible in a full day of 720. A spike is drawn as a marker on the power curve, shading as a
shaded band over its duration.

## Building by hand

```powershell
msbuild sim_ui.sln /t:Rebuild /p:Configuration=Release /p:Platform=x64
```

The four checks in `selftest\selftest.exe` compare the C++ implementation against saved Python output and
should all report `[ok]`.

## Language

The button in the lower-right corner switches the interface language. `sim_ui.exe --lang=en --auto`
starts in English directly. All interface text is defined once, in `Bi(chinese, english)` calls in the
dialog source.

## Files

| Path | Contents |
| --- | --- |
| `SimUiDlg.cpp`, `SimUiDlg.h` | Dialog, plots, injection controls, alarm log |
| `CurveCtrl.cpp`, `.h` | Curve drawing and markers |
| `Transport.cpp`, `.h` | Serial and TCP input |
| `core/DualWindowDetector.cpp` | The detector: windows, distance, alarm rule |
| `core/Calibration.cpp` | Loading and saving the calibration file |
| `core/FrameStream.cpp` | Frame parsing |
| `core/Json.cpp` | Minimal JSON reader and writer |
| `selftest/` | Comparison against saved Python output |
| `calibration/` | Calibration files shared with the Simulink model |
