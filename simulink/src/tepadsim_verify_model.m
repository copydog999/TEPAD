function report = tepadsim_verify_model(varargin)
%TEPADSIM_VERIFY_MODEL Runs three checks with the Simulink model itself (golden / dual path / injection subsystem).
%
%   ok = tepadsim_verify_model
%   ok = tepadsim_verify_model('tol', 1e-8, 'verbose', false)
%   report = tepadsim_verify_model('saveTable', fullfile(cfg.dataDir, 'model_checks.csv'))
%
%   Three checks (tepadselftest runs pure MATLAB; what runs here is "the wired model"):
%     1. replay path vs Python golden: pack the (G,P) of golden_frames.csv into a timeseries
%        and feed From Workspace, comparing pointZ / ms / ml / score / alarm tick by tick;
%     2. dual-path agreement: replayMode=0 lets the PV Plant inside the model generate the data, compared bit-exactly
%        against the pure MATLAB tepadsim_generate_scenario (the same plant_step), proving the Unit Delay
%        feedback loop has no off-by-one in its sampling timing;
%     3. injection subsystem: compare the Spike/Shade blocks of the model against the tepadsim_apply_fault result,
%        and report the first alarm delay after injection along the way (same measure as the sim_ui measurement).
%
%   The returned report is a 3x4 table: check, worst deviation, criterion, verdict.

cfg = tepadsim_setup_path();
opt = struct('tol', 1e-8, 'verbose', true, 'saveTable', '');
for k = 1:2:numel(varargin)
    fn = fieldnames(opt);
    idx = find(strcmpi(fn, varargin{k}), 1);
    if isempty(idx)
        error('tepadsim:unknownOption', 'unknown option %s', varargin{k});
    end
    opt.(fn{idx}) = varargin{k + 1};
end

calib = tepadsim_load_calibration();
golden = tepadsim_load_golden();
rows = {};

% ---- 1. model replay path vs Python golden -------------------------------
meas = [golden.g, golden.p, golden.p, ...
    repmat(calib.residualSigma, golden.nTicks, 1), golden.tick];
replayTs = timeseries(meas, golden.tick);
replayTs.Name = 'tepadsimGoldenReplay';

resG = tepadsim_simulate('replay', replayTs, 'verbose', false);
eZ = max(abs(resG.pointZ - golden.pointZ));
eMs = max(abs(resG.ms - golden.ms));
eMl = max(abs(resG.ml - golden.ml));
eScore = max(abs(resG.score - golden.score));
nAlarmDiff = sum(resG.alarm ~= golden.alarm);
worst1 = max([eZ, eMs, eMl, eScore]);
pass1 = worst1 < opt.tol && nAlarmDiff == 0;
rows(end + 1, :) = {sprintf('golden replay (%d tick)', golden.nTicks), worst1, ...
    sprintf('deviation <%g and alarm difference = 0 (actual %d)', opt.tol, nAlarmDiff), local_flag(pass1)};
local_say(opt.verbose, ['[1/3] model vs Python golden: z=%.3e ms=%.3e ml=%.3e score=%.3e ' ...
    'alarm difference=%d → %s' 10], eZ, eMs, eMl, eScore, nAlarmDiff, local_flag(pass1));

% ---- 2. in-model online generation vs pure MATLAB generation -------------
seed2 = cfg.seed + 17;
ref = tepadsim_generate_scenario('ticks', cfg.ticksPerDay, 'seed', seed2);
resP = tepadsim_simulate('ticks', cfg.ticksPerDay, 'seed', seed2, ...
    'replayMode', 0, 'verbose', false);
eG = max(abs(resP.g - ref.g));
eP = max(abs(resP.p - ref.p));
eSig = max(abs(resP.pNormal - ref.pNormal));
worst2 = max([eG, eP, eSig]);
pass2 = worst2 < opt.tol;
rows(end + 1, :) = {'dual-path agreement (PV Plant vs generate_scenario)', worst2, ...
    sprintf('max deviation of g/p/pNormal <%g', opt.tol), local_flag(pass2)};
local_say(opt.verbose, '[2/3] online generation path vs pure MATLAB: g=%.3e p=%.3e pNormal=%.3e → %s\n', ...
    eG, eP, eSig, local_flag(pass2));

% ---- 3. injection subsystem vs MATLAB-side injection ---------------------
faultSpec = struct('spikeTick', 360, 'spikeSigma', 8.0, 'shadeStart', 400, ...
    'shadeDuration', 60, 'shadeFactor', 0.30);
resF = tepadsim_simulate('ticks', cfg.ticksPerDay, 'seed', seed2, ...
    'replayMode', 0, 'verbose', false, ...
    'spikeTick', faultSpec.spikeTick, 'spikeSigma', faultSpec.spikeSigma, ...
    'shadeStart', faultSpec.shadeStart, 'shadeDuration', faultSpec.shadeDuration, ...
    'shadeFactor', faultSpec.shadeFactor);
pMatlab = tepadsim_apply_fault(ref.p, ref.sigma, ref.tick, faultSpec);
eInject = max(abs(resF.p - pMatlab));
delay = tepadsim_first_alarm_after(resF.alarm, resF.tick, faultSpec.spikeTick, 180);
pass3 = eInject < opt.tol && ~isnan(delay) && delay <= 1;
rows(end + 1, :) = {'injection subsystem (spike + shading combined)', eInject, ...
    sprintf('model/code injection deviation <%g, spike delay = %s tick', opt.tol, local_num(delay)), ...
    local_flag(pass3)};
local_say(opt.verbose, '[3/3] model injection vs apply_fault: max deviation=%.3e, spike 8σ delay=%s tick → %s\n', ...
    eInject, local_num(delay), local_flag(pass3));

ok = pass1 && pass2 && pass3;

% ---- summary table -------------------------------------------------------
report = struct();
report.pass = ok;
report.rows = rows;
report.tol = opt.tol;
report.firstAlarmTick = local_first(resF.alarm, resF.tick);
report.spikeDelayTick = delay;
report.baselineAlarm = resG.nAlarm;

if ~isempty(opt.saveTable)
    local_write_table(opt.saveTable, rows);
    if opt.verbose
        fprintf('  check results written: %s\n', opt.saveTable);
    end
end

if opt.verbose
    if ok
        fprintf('\nmodel checks all passed (golden: %s)\n', calib.summary);
    else
        fprintf('\n*** model checks FAILED ***\n');
    end
end
end

function t = local_first(alarm, tick)
idx = find(alarm, 1);
if isempty(idx)
    t = NaN;
else
    t = tick(idx);
end
end

function local_write_table(file, rows)
fid = fopen(file, 'w', 'n', 'UTF-8');
if fid < 0
    warning('tepadsim:cannotWrite', 'cannot write %s', file);
    return;
end
fprintf(fid, 'check,worst_diff,criterion,result\n');
for r = 1:size(rows, 1)
    fprintf(fid, '"%s",%.6e,"%s",%s\n', rows{r, 1}, rows{r, 2}, rows{r, 3}, rows{r, 4});
end
fclose(fid);
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
    fprintf([template], varargin{:});
end
end
