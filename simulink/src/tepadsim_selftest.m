function ok = tepadsim_selftest(varargin)
%TEPADSIM_SELFTEST Numerical/timing self-test on the simulink side; no need to open the Simulink model.
%
%   ok = tepadsim_selftest            % returns true when everything passes
%   ok = tepadsim_selftest('verbose', false)
%
%   Four checks (same idea as sim_ui/selftest, only the subject is the MATLAB implementation):
%     1. drive tepadsim_detector_step tick by tick with (G,P) from the Python golden,
%        comparing point_z / short / long / score / alarm — proving the algorithm shares the Python origin;
%     2. a normal day raises 0 alarms;
%     3. a +8σ spike alarms on its tick and a ×0.30 shading over 60 tick alarms within 180 tick;
%     4. the same sub-seed generates a bit-identical day twice (the LCG carries its own random stream, no rng dependency).
%
%   With these four green, any deviation in the Simulink model can only come from timing, not from the algorithm.

cfg = tepadsim_setup_path();
verbose = true;
for k = 1:2:numel(varargin)
    if strcmpi(varargin{k}, 'verbose')
        verbose = logical(varargin{k + 1});
    end
end

calib = tepadsim_load_calibration();
golden = tepadsim_load_golden();
ok = true;

% ---- 1. tick-by-tick comparison against the Python golden -----------------
res = tepadsim_reference_run(golden.g, golden.p, calib);
tol = 1e-8;
eZ = max(abs(res.pointZ - golden.pointZ));
eS = max(abs(res.ms - golden.ms));
eL = max(abs(res.ml - golden.ml));
eScore = max(abs(res.score - golden.score));
nAlarmDiff = sum(res.alarm ~= golden.alarm);
pass1 = (eZ < tol) && (eS < tol) && (eL < tol) && (eScore < tol) && (nAlarmDiff == 0);
ok = ok && pass1;
local_say(verbose, '[1/4] golden tick-by-tick comparison: z=%.3e ms=%.3e ml=%.3e score=%.3e alarm diff=%d → %s', ...
    eZ, eS, eL, eScore, nAlarmDiff, local_flag(pass1));

% ---- 2. no false alarms on a normal day ----------------------------------
clean = tepadsim_generate_scenario('ticks', cfg.ticksPerDay, 'seed', cfg.seed + 17);
cleanRes = tepadsim_reference_run(clean.g, clean.p, calib);
pass2 = cleanRes.nAlarm == 0;
ok = ok && pass2;
local_say(verbose, '[2/4] normal day %d tick: alarms = %d (peak score %.3f) → %s', ...
    cfg.ticksPerDay, cleanRes.nAlarm, max(cleanRes.score), local_flag(pass2));

% ---- 3. injections are detected ------------------------------------------
faultSpike = struct('spikeTick', 360, 'spikeSigma', 8.0, 'shadeStart', -1, ...
    'shadeDuration', 0, 'shadeFactor', 1.0);
pSpike = tepadsim_apply_fault(clean.p, clean.sigma, clean.tick, faultSpike);
resSpike = tepadsim_reference_run(clean.g, pSpike, calib);
delaySpike = local_delay(resSpike, 360, 180);
pass3a = ~isnan(delaySpike) && delaySpike <= 1;

faultShade = struct('spikeTick', -1, 'spikeSigma', 0, 'shadeStart', 400, ...
    'shadeDuration', 60, 'shadeFactor', 0.30);
pShade = tepadsim_apply_fault(clean.p, clean.sigma, clean.tick, faultShade);
resShade = tepadsim_reference_run(clean.g, pShade, calib);
delayShade = local_delay(resShade, 400, 180);
pass3b = ~isnan(delayShade);
ok = ok && pass3a && pass3b;
local_say(verbose, '[3/4] spike 8σ delay=%s tick, shading ×0.30/60tick delay=%s tick → %s', ...
    local_num(delaySpike), local_num(delayShade), local_flag(pass3a && pass3b));

% ---- 4. random stream is reproducible ------------------------------------
again = tepadsim_generate_scenario('ticks', cfg.ticksPerDay, 'seed', cfg.seed + 17);
maxDiff = max(abs([clean.g - again.g; clean.p - again.p; clean.sigma - again.sigma]));
pass4 = maxDiff == 0;
ok = ok && pass4;
local_say(verbose, '[4/4] max difference between two generations with the same seed = %.3e → %s', maxDiff, local_flag(pass4));

if verbose
    if ok
        fprintf('\nsimulink self-test all passed (calibration: %s)\n', calib.summary);
    else
        fprintf('\n*** simulink self-test FAILED ***\n');
    end
end
end

function d = local_delay(res, startTick, horizon)
%First alarm delay within horizon ticks after injection; NaN when nothing alarms.
%   Only look at alarms after the injection window: the normal day already verified 0 false alarms.
after = find(res.alarm & (res.tick >= startTick) & (res.tick < startTick + horizon));
d = NaN;
if ~isempty(after)
    d = res.tick(after(1)) - startTick;
end
end

function s = local_flag(pass)
if pass
    s = 'PASS';
else
    s = 'FAIL';
end
end

function s = local_num(v)
if isnan(v)
    s = 'not detected';
else
    s = sprintf('%d', v);
end
end

function local_say(verbose, template, varargin)
if verbose
    fprintf([template '\n'], varargin{:});
end
end
