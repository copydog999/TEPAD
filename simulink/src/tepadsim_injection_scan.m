function scan = tepadsim_injection_scan(varargin)
%TEPADSIM_INJECTION_SCAN Sweeps manual injection with the Simulink model: spike amplitude × shading depth.
%
%   scan = tepadsim_injection_scan
%   scan = tepadsim_injection_scan('spikeSigmas', [2 4 6 8 12], ...
%                                  'shadeFactors', [0.85 0.6 0.4 0.3 0.15], ...
%                                  'saveCsv', 'simulink/data/injection_scan.csv')
%
%   Every case is one run of "the same simulation day + the injection subsystem in the model",
%   with the data source fixed to the replay path (replayMode=1), so the only difference between cases is the fault itself
%   and delay differences are not mixed with differences in the random stream.
%
%   The reading matches the manual-injection measurements in sim_ui/README and src/simulation/experiment.py:
%     delayTicks = tick offset of the first alarm after the injection start (-1 = no alarm within horizon)
%     maxScore   = peak normalized score over the run (1.0 is the threshold line)
%
%   Returned scan
%     scan.rows    cell array, column order = CSV header
%     scan.header  header
%     scan.demo    full per-tick result of the representative case (mixed injection), for plotting
%     scan.csv     path of the written CSV (empty when nothing was written)

cfg = tepadsim_setup_path();
opt = struct( ...
    'ticks', cfg.ticksPerDay, ...
    'seed', cfg.seed + 17, ...
    'spikeTick', 360, ...
    'spikeSigmas', [2 4 6 8 12], ...
    'shadeStart', 400, ...
    'shadeDuration', 60, ...
    'shadeFactors', [0.85 0.6 0.4 0.3 0.15], ...
    'horizon', 180, ...
    'verbose', true, ...
    'saveCsv', fullfile(cfg.dataDir, 'injection_scan.csv'));
fn = fieldnames(opt);
for k = 1:2:numel(varargin)
    idx = find(strcmpi(fn, varargin{k}), 1);
    if isempty(idx)
        error('tepadsim:unknownOption', 'unknown option %s', varargin{k});
    end
    opt.(fn{idx}) = varargin{k + 1};
end

% The scenario is generated once and shared by every case (replay path = the same (g,p,sigma,tick))
scenario = tepadsim_generate_scenario('ticks', opt.ticks, 'seed', opt.seed);

header = {'case', 'kind', 'injectTick', 'magnitude', 'delayTicks', ...
    'detected', 'nAlarm', 'maxScore'};
rows = {};

% ---- baseline (no injection) ---------------------------------------------
base = tepadsim_simulate('replay', scenario.replay, 'verbose', false);
rows(end + 1, :) = {'baseline', 'none', 0, '-', -1, false, base.nAlarm, base.maxScore};
local_say(opt.verbose, '  baseline      none   %d alarms, peak score %.3f\n', ...
    base.nAlarm, base.maxScore);

% ---- spike ---------------------------------------------------------------
for i = 1:numel(opt.spikeSigmas)
    sigma = opt.spikeSigmas(i);
    res = tepadsim_simulate('replay', scenario.replay, 'verbose', false, ...
        'spikeTick', opt.spikeTick, 'spikeSigma', sigma);
    delay = local_delay(res.alarm, res.tick, opt.spikeTick, opt.horizon);
    rows(end + 1, :) = {sprintf('spike %gσ', sigma), 'spike', opt.spikeTick, ...
        sprintf('%gσ', sigma), delay, delay >= 0, res.nAlarm, res.maxScore};
    local_say(opt.verbose, '  spike %-8s tick=%-4d delay=%-4s alarms %d, peak %.3f\n', ...
        sprintf('%gσ', sigma), opt.spikeTick, local_delayText(delay), res.nAlarm, res.maxScore);
end

% ---- shading -------------------------------------------------------------
for i = 1:numel(opt.shadeFactors)
    factor = opt.shadeFactors(i);
    res = tepadsim_simulate('replay', scenario.replay, 'verbose', false, ...
        'shadeStart', opt.shadeStart, 'shadeDuration', opt.shadeDuration, ...
        'shadeFactor', factor);
    delay = local_delay(res.alarm, res.tick, opt.shadeStart, opt.horizon);
    rows(end + 1, :) = {sprintf('shading ×%.2f', factor), 'shading', opt.shadeStart, ...
        sprintf('×%.2f/%d', factor, opt.shadeDuration), delay, delay >= 0, ...
        res.nAlarm, res.maxScore};
    local_say(opt.verbose, '  shading %-8s tick=%-4d delay=%-4s alarms %d, peak %.3f\n', ...
        sprintf('×%.2f', factor), opt.shadeStart, local_delayText(delay), res.nAlarm, res.maxScore);
end

% ---- mixed (spike + shading, equivalent to clicking both buttons on the UI) ---
resMixed = tepadsim_simulate('replay', scenario.replay, 'verbose', false, ...
    'spikeTick', opt.spikeTick, 'spikeSigma', 8, ...
    'shadeStart', opt.shadeStart, 'shadeDuration', opt.shadeDuration, 'shadeFactor', 0.30);
delayMixed = local_delay(resMixed.alarm, resMixed.tick, opt.spikeTick, opt.horizon);
rows(end + 1, :) = {'mixed 8σ+×0.30', 'mixed', opt.spikeTick, '8σ + ×0.30/60', ...
    delayMixed, delayMixed >= 0, resMixed.nAlarm, resMixed.maxScore};
local_say(opt.verbose, '  mixed(spike 8σ@%d + shading ×0.30@%d): first alarm delay=%s, %d alarms, peak %.3f\n', ...
    opt.spikeTick, opt.shadeStart, local_delayText(delayMixed), resMixed.nAlarm, resMixed.maxScore);

scan.rows = rows;
scan.header = header;
scan.demo = resMixed;
scan.base = base;
scan.scenario = struct('ticks', opt.ticks, 'seed', opt.seed);
scan.spikeTick = opt.spikeTick;
scan.shadeStart = opt.shadeStart;
scan.shadeDuration = opt.shadeDuration;
scan.horizon = opt.horizon;
% Sweep grid used for plotting (so amplitudes need not be parsed back out of CSV string columns)
scan.spikeSigmas = opt.spikeSigmas;
scan.shadeFactors = opt.shadeFactors;

if ~isempty(opt.saveCsv)
    if exist(fileparts(opt.saveCsv), 'dir') ~= 7
        mkdir(fileparts(opt.saveCsv));
    end
    local_write_csv(opt.saveCsv, header, rows);
    scan.csv = opt.saveCsv;
    if opt.verbose
        fprintf('  scan results written: %s\n', opt.saveCsv);
    end
else
    scan.csv = '';
end
end

function d = local_delay(alarm, tick, startTick, horizon)
% First alarm offset within horizon ticks after the injection start; -1 when nothing alarms (the CSV wants an integer).
d = tepadsim_first_alarm_after(alarm, tick, startTick, horizon);
if isnan(d)
    d = -1;
end
end

function s = local_delayText(d)
if d < 0
    s = 'not detected';
else
    s = sprintf('%d', d);
end
end

function local_write_csv(file, header, rows)
fid = fopen(file, 'w', 'n', 'UTF-8');
if fid < 0
    warning('tepadsim:cannotWrite', 'cannot write %s', file);
    return;
end
fprintf(fid, '%s\n', strjoin(header, ','));
for r = 1:size(rows, 1)
    fprintf(fid, '"%s",%s,%d,"%s",%d,%s,%d,%.6f\n', rows{r, 1}, rows{r, 2}, ...
        rows{r, 3}, rows{r, 4}, rows{r, 5}, local_tf(rows{r, 6}), ...
        rows{r, 7}, rows{r, 8});
end
fclose(fid);
end

function s = local_tf(flag)
if logical(flag)
    s = 'true';
else
    s = 'false';
end
end

function local_say(verbose, template, varargin)
if verbose
    % The template already carries newlines; do not add one, otherwise every log line is separated by a blank line.
    fprintf(template, varargin{:});
end
end
