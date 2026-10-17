# The model

This page explains what the detector computes, in the order the code computes it. Every formula below
appears in `src/model/` and `src/detection/`, and the file names are given at the end.

## 1. Irradiance and power on a normal day

Irradiance is bounded and unevenly distributed during the day, so we describe it with a Beta distribution.
Power is then modelled as a Gaussian whose mean follows the irradiance and whose spread grows with the
mean, because a plant that is producing near its rated output fluctuates more than one producing a little.

With normalized irradiance $g\in[0,1]$ and normalized power $p\in[0,1]$:

$$
G \sim \mathrm{Beta}(\alpha,\beta),
\qquad
P \mid G=g \sim \mathcal N\big(\mu(g),\ \sigma^2(g)\big)
$$

$$
\mu(g)=\min(\eta g,\ 1),
\qquad
\sigma(g)=\sigma_0\,\mu(g)+\sigma_{\min}
$$

The mean is capped at 1 because normalized power cannot exceed the rated output. The three remaining
quantities have direct physical readings:

- $\eta$ is the conversion efficiency, the slope of power against irradiance.
- $\sigma_0$ says how quickly the fluctuation grows with output.
- $\sigma_{\min}$ is the noise that remains at low output.

Together with the two shape parameters $\alpha$ and $\beta$, a window of a day is described by five
numbers, $\theta=[\alpha,\beta,\eta,\sigma_0,\sigma_{\min}]$.

One property matters later: the level sets of this joint distribution are ellipses. That is why distances
to the normal cloud can be measured with the Mahalanobis distance, which is the natural metric for an
elliptical cloud.

## 2. Estimating the five numbers from one window

A window is a short stretch of samples. Inside it, the five numbers are estimated in three steps, each a
closed-form fit rather than an iterative optimiser.

**Shape parameters.** Match the sample mean $\bar g$ and variance of the irradiance to the Beta moments:

$$
c=\frac{\bar g(1-\bar g)}{\max(\widehat{\mathrm{Var}}(g),\ 10^{-6})}-1,
\qquad
\alpha=\bar g\,c,
\qquad
\beta=(1-\bar g)c
$$

The $10^{-6}$ floor stops the estimate from exploding when the window is almost constant. If $c\le 0$ or
$\bar g$ is outside $(0,1)$, the five numbers for that window are recorded as `nan`.

**Efficiency.** A least-squares fit of $p$ against $g$ gives $\eta$.

**Noise.** With $\hat\mu_i=\min(\eta g_i,1)$ and $r_i=p_i-\hat\mu_i$, a least-squares fit of $|r_i|$ against
$\hat\mu_i$ gives the intercept $\sigma_{\min}$ and the slope $\sigma_0$. Fewer than four samples, or an
almost constant $\hat\mu$, again gives `nan`.

## 3. Turning a day into one vector

A day is cut into ten overlapping windows of 30 samples with a stride of 10. Each window contributes its
five numbers, and the ten five-number blocks are concatenated, so one day becomes a vector of 50 numbers.
Windows that failed to fit are filled with the median of the corresponding coordinate over normal days.

The code also fits AR(1) and AR(2) models to the residuals. These are kept as diagnostics; they take no
part in the decision.

## 4. First check: distance from the normal cloud

From normal days we estimate a mean vector $\mu_0$ and a covariance matrix $\Sigma_0$, with $10^{-6}I$ added
to the diagonal so the inverse is well defined. A day with features $x$ is scored by

$$
d(x)=\sqrt{(x-\mu_0)^{\top}\Sigma_0^{-1}(x-\mu_0)}
$$

The threshold is the 95th percentile of $d$ measured on the normal days that built $\mu_0$ and $\Sigma_0$.
The first check fires when $d$ exceeds it.

Because $d$ accounts for the correlations between the 50 numbers, it reacts to combinations that never
occur on normal days even when each individual number is unremarkable.

## 5. Second check: disagreement with an independent power estimate

The first check only sees the relation between measured irradiance and measured power. Shading a
pyranometer and the modules at the same time lowers both, the relation still looks normal, and the fault
goes unnoticed. The second check therefore compares measured power with a power estimate that does not
come from the same sensor pair.

The estimate is taken from the best available source, in this order:

| Order | Source | Used for |
| --- | --- | --- |
| 1 | Modelled or theoretical power shipped with the dataset | Ding |
| 2 | A healthy twin inverter | La Réunion |
| 3 | The fitted relation $\min(\eta g,1)$ | DS226, and any dataset without an external estimate |

The residual is $r_t=p_t-p^{\mathrm{ref}}_t$. Negative residuals deeper than
$\tau_{\mathrm{res}}=2\,\mathrm{std}(r)$ on normal days are treated as drops, and five numbers summarise
the residual curve:

| Feature | What it measures | Fires when |
| --- | --- | --- |
| $r_{\min}$ | deepest drop | below its 1st percentile |
| $\Delta r_{\min}$ | steepest fall | below its 1st percentile |
| $f_{\mathrm{neg}}$ | share of samples below $-\tau_{\mathrm{res}}$ | above its 99th percentile |
| $T_{\mathrm{neg}}$ | longest unbroken run below $-\tau_{\mathrm{res}}$ | reported, not thresholded |
| $\mathrm{corr}$ | correlation between measured and estimated power | below its 1st percentile |

The second check fires if any of the three thresholded features crosses its limit:

$$
J=(r_{\min}<\tau_1)\ \lor\ (\Delta r_{\min}<\tau_2)\ \lor\ (f_{\mathrm{neg}}>\tau_3)\ \lor\ (\mathrm{corr}<\tau_5)
$$

## 6. Combining the two checks

A day is reported as faulty if either check fires, and normal otherwise.

| First check | Second check | Result |
| --- | --- | --- |
| fires | any | faulty |
| does not fire | fires | faulty |
| does not fire | does not fire | normal |

The second check alone is not reliable: spikes, inverter faults and snowy days also leave deep residual
dips, so it would flag many faults of the wrong kind. Applying the first check first removes the faults
that already show up in the irradiance-power relation, and the second check then only has to catch what
remains. The two checks are not independent detectors; they are two questions asked of the same model.

## 7. Settings

All of these live in `src/config.py`.

| Setting | Value | Meaning |
| --- | --- | --- |
| `VARIANCE_FLOOR` | $10^{-6}$ | Floor on the irradiance variance in the shape estimate |
| `WINDOW`, `STEP`, `N_WINDOWS` | 30, 10, 10 | Window length, stride, windows per day |
| `DAYLIGHT_QUANTILE` | 0.05 | Samples below this irradiance quantile are dropped |
| `MIN_DAY_POINTS` | 10 | Shortest day curve that is kept |
| `STAGE1_QUANTILE` | 95 | Threshold percentile of the first check |
| `STAGE2_Q_LOW`, `STAGE2_Q_HIGH` | 1, 99 | Threshold percentiles of the second check |
| `RESID_TAU_K` | 2.0 | Multiplier defining a residual drop |
| `SEED` | 42 | Random seed |

Simulation settings are in the same file, under the simulation section.

## 8. Where this is implemented

| What | File |
| --- | --- |
| Five-number estimation, mean and spread functions | `src/model/probabilistic.py` |
| Window flattening, local residuals, AR fits | `src/model/param_estimation.py` |
| Choice of the external power estimate | `src/model/reference.py` |
| Residual drops and the five features | `src/model/residual.py` |
| Distance and threshold of the first check | `src/detection/stage1.py` |
| Criterion of the second check | `src/detection/stage2.py` |
| Rule that combines them | `src/detection/cascade.py`, `run_two_level` |
| Distance updated one sample at a time, for the live and Simulink versions | `src/model/mahalanobis.py` |
