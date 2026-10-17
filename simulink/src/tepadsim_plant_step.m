function [meas, stateOut] = tepadsim_plant_step(par, stateIn)
%TEPADSIM_PLANT_STEP Per-tick data generator for a PV plant (Beta cloud shading + PVWatts + heteroscedastic noise).
%
%   [meas, stateOut] = tepadsim_plant_step(par, stateIn)
%
%   Structure corresponds step by step with Python src/simulation/model.py::simulate_normal_day:
%     1) 12-hour clear-sky envelope clear_sky = 1000*sin(phase), phase = pi*tick/(n-1)
%     2) cloud shading factor ~ Beta(alpha,beta), then a first-order AR(0.75) inertia for cloud persistence
%     3) ambient temperature = base + 2*sin(phase-0.4), cell temperature = ambient + (45-20)/800 * G
%     4) PVWatts normalized AC power pNormal (temperature derating + 0.96 system loss, clipped to [0,1])
%     5) heteroscedastic measurement noise sigma = noise_scale*(sigma0*pNormal + sigma_min)
%     6) p = clip(pNormal + N(0,sigma), 0, 1), g = clip(irradiance/1000, 0, 1)
%
%   Random numbers do not use the MATLAB/Simulink global RNG; instead this carries a 32-bit Lehmer LCG + Box-Muller
%   normal + Marsaglia-Tsang Gamma, with all state kept in stateIn:
%     * independent of the Simulink "Random Numbers" configuration and the MATLAB workspace rng state;
%     * bit-reproducible for a given seed, which keeps regression comparisons easy;
%     * NumPy uses PCG64, so bit-exact alignment was never possible; only the distribution family and structure are matched here.
%     Per-tick numerical ground-truth verification goes through the replay path (tepadsim_verify_golden),
%     which consumes the golden_frames.csv exported by Python and is unaffected by the random stream of this file.
%
%   Inputs
%     par     : 9 elements [alpha beta eta sigma0 sigma_min nTicks temperatureBase noiseScale seed]
%     stateIn : 3 elements [nSeen; prevPersistent; rngState]
%   Outputs
%     meas     : 5x1 [g; p; pNormal; sigma; tick] (tick starts at 0, matching the Python index)
%     stateOut : 3x1 state for the next tick
%
%   The output is a column vector rather than a 1x5 row vector: MATLAB Function treats Mx1 as a 1-D vector
%   and 1xN as a matrix, and the Demux/Switch downstream can only split scalar ports off a 1-D vector.
%
%   This file is also the script source of the PV Plant MATLAB Function block in the .slx,
%   so it uses only the syntax subset MATLAB Function supports (no error/assert/toolbox functions/recursion).
% -------------------------------------------------------------------------

par = par(:);
stateIn = stateIn(:);

alphaCloud = par(1);
betaCloud = par(2);
eta = par(3);
sigma0 = par(4);
sigmaMin = par(5);
nTicks = par(6);
temperatureBase = par(7);
noiseScale = par(8);
pvGamma = -0.004;              % default temperature coefficient of model.py::pvwatts_power

nSeen = stateIn(1);
persistentPrev = stateIn(2);
rngState = stateIn(3);

% ---- clear-sky envelope and cloud shading --------------------------------
phase = pi * nSeen / local_max(nTicks - 1.0, 1.0);
clearSky = 1000.0 * sin(phase);
[cloud, rngState] = local_beta(alphaCloud, betaCloud, rngState);

if nSeen <= 0.0
    % Matches Python persistent[0] = cloud[0]: the first frame skips the AR recursion.
    persistentNow = cloud;
else
    persistentNow = 0.75 * persistentPrev + 0.25 * cloud;
end
if persistentNow < 0.0
    persistentNow = 0.0;
elseif persistentNow > 1.0
    persistentNow = 1.0;
end

irradiance = clearSky * persistentNow;
temperature = temperatureBase + 2.0 * sin(phase - 0.4);
cellTemperature = temperature + ((45.0 - 20.0) / 800.0) * irradiance;
pdc = eta * (irradiance / 1000.0) * (1.0 + pvGamma * (cellTemperature - 25.0));
pNormal = 0.96 * pdc;
if pNormal < 0.0
    pNormal = 0.0;
elseif pNormal > 1.0
    pNormal = 1.0;
end

% ---- heteroscedastic measurement noise ------------------------------------
sigma = noiseScale * (sigma0 * pNormal + sigmaMin);
if sigma < 1e-9
    sigma = 1e-9;
end
[noise, rngState] = local_normal(rngState);
p = pNormal + sigma * noise;
if p < 0.0
    p = 0.0;
elseif p > 1.0
    p = 1.0;
end

g = irradiance / 1000.0;
if g < 0.0
    g = 0.0;
elseif g > 1.0
    g = 1.0;
end

meas = [g; p; pNormal; sigma; nSeen];
stateOut = [nSeen + 1.0; persistentNow; rngState];
end

% =========================================================================
% random number and numerical helpers
% =========================================================================

function [u, state] = local_uniform(state)
%Lehmer/Minstd LCG: state = mod(state*16807, 2^31-1), all double, no overflow risk.
state = mod(state * 16807.0, 2147483647.0);
if state < 1.0
    state = 1.0;
end
u = state / 2147483647.0;
if u >= 1.0
    u = 1.0 - 1e-12;
end
end

function [z, state] = local_normal(state)
%Box-Muller, taking only the cos branch (no pair caching, simpler state).
[u1, state] = local_uniform(state);
[u2, state] = local_uniform(state);
z = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2);
end

function [x, state] = local_gamma(shape, state)
%Marsaglia-Tsang Gamma sampling; for shape<1, sample with shape+1 and multiply by U^(1/shape) to boost it,
%   using a branch instead of recursion (MATLAB Function has limited recursion support).
boosted = false;
shapeEff = shape;
if shape < 1.0
    boosted = true;
    shapeEff = shape + 1.0;
end
d = shapeEff - 1.0 / 3.0;
c = 1.0 / sqrt(9.0 * d);
x = 0.0;
accepted = false;
while ~accepted
    [z, state] = local_normal(state);
    v = 1.0 + c * z;
    if v > 0.0
        [u, state] = local_uniform(state);
        if log(u) < 0.5 * z * z + d - d * v + d * log(v)
            x = d * v * v * v;
            accepted = true;
        end
    end
end
if boosted
    [u2, state] = local_uniform(state);
    x = x * u2^(1.0 / shape);
end
end

function [b, state] = local_beta(a, bb, state)
%Beta(a,bb) = X/(X+Y), X~Gamma(a), Y~Gamma(bb), same distribution family as rng.beta.
[x, state] = local_gamma(a, state);
[y, state] = local_gamma(bb, state);
denom = x + y;
if denom <= 1e-300
    b = 0.5;
else
    b = x / denom;
end
if b < 0.0
    b = 0.0;
elseif b > 1.0
    b = 1.0;
end
end

function y = local_max(x, m)
y = max(x, m);
end
