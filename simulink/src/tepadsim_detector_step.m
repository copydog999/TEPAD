function [out, stateOut] = tepadsim_detector_step(g, p, calib, stateIn)
%TEPADSIM_DETECTOR_STEP One-tick advance of the TEPAD dual-window detector (Simulink external-state version).
%
%   [out, stateOut] = tepadsim_detector_step(g, p, calib, stateIn)
%
%   Numerical semantics aligned line by line with two existing implementations:
%     * Python  src/simulation/model.py::DualWindowDetector.step (including estimate_theta /
%               CausalParameterEstimator / _features_from_params)
%     * C++     sim_ui/core/DualWindowDetector.cpp::Step
%   The only difference is that state is not kept in object members/persistent variables; the two sliding-window buffers, the EWMA parameters and
%   the ready flags are packed into an explicit state vector, which Simulink can feed back with a single Unit Delay,
%   so there is no hidden state and no cross-talk between simulation runs.
%
%   Three-channel OR alarm:
%     channel 1  point residual z  = |p - eta_ref * g| / max(residual_sigma, 0.025*max(g,0.1))
%     channel 2  short-window Mahalanobis (L=10, causal parameter estimate with EWMA α=0.1)
%     channel 3  long-window Mahalanobis (L=60, same as above)
%   score = max(z/tau_z, ms/tau_short, ml/tau_long); alarm = any channel above its threshold.
%
%   Inputs
%     g, p    : normalized irradiance (0~1) and normalized AC power (0~1) of the current tick
%     calib   : packed 91-element calibration constant (row or column vector both accepted, see (:) below),
%               layout below
%     stateIn : 153-element state vector, layout below
%   Outputs
%     out      : 7x1 [score; alarm; pointZ; ms; ml; shortReady; longReady]
%     stateOut : 153x1 state for the next tick
%
%   Why always "column vectors" and never 1xN row vectors: a MATLAB Function block treats Mx1 as a
%   1-D vector signal and 1xN (N>1) as a matrix signal; Demux can only split scalar ports out of a vector,
%   a matrix yields [1x1] matrices and every downstream scalar port rejects them (verified the hard way).
%
%   calib layout (indices from 1)
%     1  eta_reference        2  residual_sigma       3  residual_threshold
%     4  tau_short            5  tau_long             6  z_clip
%     7  sigma_floor_rate     8  g_floor              9  short_window
%     10 long_window          11 ewma_alpha
%     12:16 short center      17:21 short scale       22:26 short standardized_center
%     27:51 short inverse_covariance (reshape(5,5), column-major)
%     52:56 long center       57:61 long scale        62:66 long standardized_center
%     67:91 long inverse_covariance (reshape(5,5), column-major)
%
%   stateIn layout
%     1          nSeen (frames fed in before this tick)
%     2:11       gShort (old→new), 12:21 pShort
%     22:81      gLong , 82:141 pLong
%     142:146    thetaShort (raw 5 parameters, not log features)
%     147:151    thetaLong
%     152        hasShort (0/1, whether EWMA is initialized)
%     153        hasLong
%
%   Note: this file is also the script source of the MATLAB Function block in the .slx (tepadsim_build_model writes it
%   verbatim with fileread), so only the syntax subset supported by MATLAB Function is allowed:
%   no error/assert/fprintf, no toolbox functions.
% -------------------------------------------------------------------------

SHORT_LEN = 10;
LONG_LEN = 60;

% Whatever comes in, row or column, is squashed to a column here so the internals need not care about concatenation direction.
calib = calib(:);
stateIn = stateIn(:);

% ---- unpack the calibration constants -------------------------------------
etaRef = calib(1);
residualSigma = calib(2);
residualThreshold = calib(3);
tauShort = calib(4);
tauLong = calib(5);
zClip = calib(6);
sigmaFloorRate = calib(7);
gFloor = calib(8);
ewmaAlpha = calib(11);

shortCenter = calib(12:16);
shortScale = calib(17:21);
shortZCenter = calib(22:26);
shortInvCov = reshape(calib(27:51), 5, 5);
longCenter = calib(52:56);
longScale = calib(57:61);
longZCenter = calib(62:66);
longInvCov = reshape(calib(67:91), 5, 5);

% ---- unpack the state ----------------------------------------------------
nSeen = stateIn(1);
gShort = stateIn(2:11);
pShort = stateIn(12:21);
gLong = stateIn(22:81);
pLong = stateIn(82:141);
thetaShort = stateIn(142:146);
thetaLong = stateIn(147:151);
hasShort = stateIn(152) > 0.5;
hasLong = stateIn(153) > 0.5;

% ---- channel 1: point residual z -----------------------------------------
sigmaPoint = local_max(residualSigma, sigmaFloorRate * local_max(g, gFloor));
pointZ = local_abs(p - etaRef * g) / sigmaPoint;

% ---- sliding-window advance (FIFO shift, equivalent to Python append + del [0]) ---
gShort = [gShort(2:SHORT_LEN); g];
pShort = [pShort(2:SHORT_LEN); p];
gLong = [gLong(2:LONG_LEN); g];
pLong = [pLong(2:LONG_LEN); p];
nNext = nSeen + 1;

% ---- channels 2/3: causal parameter estimate + Mahalanobis ----------------
shortReady = nNext >= SHORT_LEN;
longReady = nNext >= LONG_LEN;

ms = 0.0;
ml = 0.0;

if shortReady
    [estimate, valid] = local_estimate_theta(gShort, pShort);
    if valid
        if hasShort
            thetaShort = ewmaAlpha * estimate + (1.0 - ewmaAlpha) * thetaShort;
        else
            thetaShort = estimate;
            hasShort = true;
        end
        ms = local_mahalanobis(thetaShort, shortCenter, shortScale, shortZCenter, ...
                               shortInvCov, zClip);
    else
        % Invalid estimate: same as CausalParameterEstimator.step returning None in Python;
        % neither is the EWMA updated nor is this tick considered ready (the score stays 0).
        shortReady = false;
    end
end

if longReady
    [estimate, valid] = local_estimate_theta(gLong, pLong);
    if valid
        if hasLong
            thetaLong = ewmaAlpha * estimate + (1.0 - ewmaAlpha) * thetaLong;
        else
            thetaLong = estimate;
            hasLong = true;
        end
        ml = local_mahalanobis(thetaLong, longCenter, longScale, longZCenter, ...
                               longInvCov, zClip);
    else
        longReady = false;
    end
end

% ---- three-channel OR alarm + normalized score ---------------------------
alarm = (pointZ > residualThreshold) || (ms > tauShort) || (ml > tauLong);
normalizedPoint = pointZ / local_max(residualThreshold, 1e-12);
normalizedShort = ms / local_max(tauShort, 1e-12);
normalizedLong = ml / local_max(tauLong, 1e-12);
score = local_max(local_max(normalizedPoint, normalizedShort), normalizedLong);

out = [score; local_double(alarm); pointZ; ms; ml; ...
       local_double(shortReady); local_double(longReady)];

stateOut = [nNext; gShort; pShort; gLong; pLong; thetaShort; thetaLong; ...
            local_double(hasShort); local_double(hasLong)];
end

% =========================================================================
% The local functions below correspond one-to-one with the same-named sections of model.py / DualWindowDetector.cpp.
% =========================================================================

function y = local_abs(x)
y = abs(x);
end

function z = local_max(x, y)
z = max(x, y);
end

function d = local_double(flag)
% Convert logical to double so the state vector/output bus always carries double.
if flag
    d = 1.0;
else
    d = 0.0;
end
end

function [params, valid] = local_estimate_theta(g, p)
%LOCAL_ESTIMATE_THETA Moment estimation + OLS correction, aligned with model.py::estimate_theta.
%   alpha/beta come from the first two moments of the Beta distribution; eta is the OLS slope of p on g (centred on the clipped
%   mean_g, matching Python's in-place rewrite of mean_g); sigma0/sigma_min come from regressing |p - mu| converted to a
%   Gaussian σ onto mu.
%
%   Python filters by isfinite before summing; here it becomes "set non-finite terms to 0 + count separately", because all
%   statistics are additive, both give exactly the same result, and array sizes stay fixed (MATLAB Function does not accept
%   variable-length outputs).
valid = true;
cnt = 0;
gv = g;
pv = p;
for i = 1:numel(g)
    if isfinite(g(i)) && isfinite(p(i))
        cnt = cnt + 1;
    else
        gv(i) = 0.0;
        pv(i) = 0.0;
    end
end
n = cnt;
if n < 4
    valid = false;
    params = [nan; nan; nan; nan; nan];
    return;
end

sumG = sum(gv);
sumP = sum(pv);
sumGG = sum(gv .* gv);
sumGP = sum(gv .* pv);
meanG = sumG / n;
meanP = sumP / n;

if meanG < 1e-5
    meanG = 1e-5;
elseif meanG > 1.0 - 1e-5
    meanG = 1.0 - 1e-5;
end

varG = (sumGG - n * meanG * meanG) / local_max(n - 1, 1);
concentration = meanG * (1.0 - meanG) / local_max(varG, 1e-8) - 1.0;
if concentration < 1e-4
    concentration = 1e-4;
elseif concentration > 1e5
    concentration = 1e5;
end
alpha = meanG * concentration;
if alpha < 1e-4
    alpha = 1e-4;
elseif alpha > 1e5
    alpha = 1e5;
end
beta = (1.0 - meanG) * concentration;
if beta < 1e-4
    beta = 1e-4;
elseif beta > 1e5
    beta = 1e5;
end

varCenteredG = sumGG - n * meanG * meanG;
if varCenteredG > 1e-12
    eta = (sumGP - n * meanG * meanP) / varCenteredG;
else
    eta = 0.0;
end
if eta < 0.0
    eta = 0.0;
elseif eta > 2.0
    eta = 2.0;
end

mu = local_min(eta * gv, 1.0);
absResidual = local_abs(pv - mu) * 1.25331413731550025519;   % sqrt(pi/2)

meanMu = sum(mu) / n;
meanAr = sum(absResidual) / n;
varMu = sum(mu .* mu) - n * meanMu * meanMu;
if varMu > 1e-12
    sigma0 = (sum(mu .* absResidual) - n * meanMu * meanAr) / varMu;
    sigmaMin = meanAr - sigma0 * meanMu;
else
    sigma0 = 0.0;
    sigmaMin = meanAr;
end
if sigma0 < 0.0
    sigma0 = 0.0;
elseif sigma0 > 2.0
    sigma0 = 2.0;
end
if sigmaMin < 1e-6
    sigmaMin = 1e-6;
elseif sigmaMin > 2.0
    sigmaMin = 2.0;
end

params = [alpha; beta; eta; sigma0; sigmaMin];
if ~(isfinite(params(1)) && isfinite(params(2)) && isfinite(params(3)) && ...
     isfinite(params(4)) && isfinite(params(5)))
    valid = false;
end
end

function y = local_min(a, b)
% Covers both the scalar and the vector case (eta*gv is a 1xN row vector, b is a scalar).
if numel(a) == 1
    y = min(a, b);
else
    y = a;
    for i = 1:numel(a)
        if a(i) > b
            y(i) = b;
        end
    end
end
end

function features = local_features(params)
%LOCAL_FEATURES Matches model.py::_features_from_params: alpha/beta/sigma0/sigma_min
%   are log-transformed (indices 1,2,4,5), eta keeps its value.
features = params;
idx = [1 2 4 5];
for k = 1:4
    base = local_max(features(idx(k)), 1e-8);
    features(idx(k)) = log(base);
end
end

function score = local_mahalanobis(params, center, scale, zCenter, invCov, zClip)
%LOCAL_MAHALANOBIS Robust standardization + clipping + sqrt(d' Σ⁻¹ d), aligned with MahalanobisReference.score.
features = local_features(params);
z = (features - center) ./ scale;
for i = 1:5
    if z(i) < -zClip
        z(i) = -zClip;
    elseif z(i) > zClip
        z(i) = zClip;
    end
end
d = z - zCenter;
quad = d.' * invCov * d;
if quad < 0.0
    quad = 0.0;
end
score = sqrt(quad);
end
