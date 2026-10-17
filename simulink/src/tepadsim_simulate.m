function res = tepadsim_simulate(varargin)
%TEPADSIM_SIMULATE Runs the Simulink model once and returns the per-tick records (scenario/injection are all parameters).
%
%   res = tepadsim_simulate                              % default: replay a normal day, no injection
%   res = tepadsim_simulate('spikeTick', 360, 'spikeSigma', 8)
%   res = tepadsim_simulate('shadeStart', 400, 'shadeDuration', 60, 'shadeFactor', 0.3)
%   res = tepadsim_simulate('replayMode', 0)             % use the in-model online PV model
%   res = tepadsim_simulate('replay', goldenTs)          % feed a timeseries directly (golden check)
%
%   Options
%     ticks / seed / replayMode  scenario parameters (replayMode~=0 takes the replay path)
%     replay                     replaces the replay data source directly (timeseries, 5 columns)
%     spikeTick spikeSigma shadeStart shadeDuration shadeFactor
%                              injection parameters; pass a negative spikeTick/shadeStart to disable a channel
%     model                    model file path, default simulink/model/tepadsim_pv_tepad.slx;
%                              built automatically with tepadsim_build_model when missing
%     verbose                  whether to print a one-line summary (default true)
%
%   Returned res
%     res.time tick g p pNormal score alarm pointZ ms ml   equal-length column vectors
%     res.nAlarm        total number of alarms
%     res.maxScore      peak score (after threshold normalization 1.0 is the alarm line)
%     res.model         model name
%     res.log           raw To Workspace matrix, column order = Log Mux order
%                       [tick g p pNormal score alarm pointZ ms ml]
%     res.fault         injection parameters in effect for this run
%     res.scenario      scenario parameters in effect for this run

cfg = tepadsim_setup_path();


opt = struct('ticks', cfg.ticksPerDay, 'seed', cfg.seed, 'replayMode', 1, ...
    'replay', [], 'model', cfg.modelFile, 'verbose', true, ...
    'spikeTick', cfg.fault.spikeTick, 'spikeSigma', cfg.fault.spikeSigma, ...
    'shadeStart', cfg.fault.shadeStart, 'shadeDuration', cfg.fault.shadeDuration, ...
    'shadeFactor', cfg.fault.shadeFactor);
for k = 1:2:numel(varargin)
    % Option names are case-insensitive (isfield is case-sensitive itself; a plain lower comparison would classify replayMode as unknown).
    fn = fieldnames(opt);
    idx = find(strcmpi(fn, varargin{k}), 1);
    if isempty(idx)
        error('tepadsim:unknownOption', 'unknown option %s', varargin{k});
    end
    opt.(fn{idx}) = varargin{k + 1};
end

% ---- model in place (script generated, so topology and code share one source) ---
if exist(opt.model, 'file') ~= 2
    tepadsim_build_model('force', true);
    opt.model = cfg.modelFile;
end
modelName = file_parts(opt.model, 'name');
if ~bdIsLoaded(modelName)
    load_system(opt.model);
end

% ---- the three workspace structs -----------------------------------------
scenario = tepadsim_generate_scenario('ticks', opt.ticks, 'seed', opt.seed);
scenario.replayMode = opt.replayMode;
if ~isempty(opt.replay)
    scenario.replay = opt.replay;
    scenario.replayMode = 1;
    scenario.ticks = numel(opt.replay.Time);
end
fault = tepadsim_make_fault('spikeTick', opt.spikeTick, 'spikeSigma', opt.spikeSigma, ...
    'shadeStart', opt.shadeStart, 'shadeDuration', opt.shadeDuration, ...
    'shadeFactor', opt.shadeFactor);
calib = tepadsim_load_calibration();

assignin('base', 'tepadsimScenario', scenario);
assignin('base', 'tepadsimFault', fault);
assignin('base', 'tepadsimCalib', calib);
evalin('base', 'clear tepadsimLog');

set_param(modelName, 'StopTime', num2str(scenario.ticks - 1), ...
    'FixedStep', '1', 'Solver', 'FixedStepDiscrete');

% ---- run the simulation --------------------------------------------------
% Newer sim does not write to the base workspace when called without output arguments, so capture the SimulationOutput explicitly,
% then mirror tepadsimLog into the workspace by hand so plots can be drawn straight from the command window.
simOut = sim(modelName, 'ReturnWorkspaceOutputs', 'on');
log = simOut.('tepadsimLog');
assignin('base', 'tepadsimLog', log);

% To Workspace(Array) does not prepend a time column in practice: one row is the 9 Log Mux signals.
% Just in case (behaviour differs across versions) both layouts are readable; decide on the column count first.
if size(log, 2) == 10
    timeCol = log(:, 1);
    data = log(:, 2:end);
else
    data = log;
    timeCol = [];
end

res.tick = data(:, 1);
res.g = data(:, 2);
res.p = data(:, 3);
res.pNormal = data(:, 4);
res.score = data(:, 5);
res.alarm = data(:, 6) > 0.5;
res.pointZ = data(:, 7);
res.ms = data(:, 8);
res.ml = data(:, 9);
if isempty(timeCol)
    res.time = res.tick * cfg.tickMinutes;   % 1 tick = 1 minute
else
    res.time = timeCol;
end
res.nAlarm = sum(res.alarm);
res.maxScore = max(res.score);
res.model = modelName;
res.log = log;
res.fault = fault;
res.scenario = struct('ticks', scenario.ticks, 'seed', scenario.seed, ...
    'replayMode', scenario.replayMode);
res.firstAlarmTick = local_first_tick(res.alarm, res.tick);

if opt.verbose
    if isnan(res.firstAlarmTick)
        fprintf('[sim] %d tick, %d alarms, peak score %.3f, data source=%s\n', ...
            scenario.ticks, res.nAlarm, res.maxScore, local_source(scenario.replayMode));
    else
        fprintf('[sim] %d tick, %d alarms, first alarm tick=%d, peak score %.3f, data source=%s\n', ...
            scenario.ticks, res.nAlarm, res.firstAlarmTick, res.maxScore, ...
            local_source(scenario.replayMode));
    end
end
end

function t = local_first_tick(alarm, tick)
idx = find(alarm, 1);
if isempty(idx)
    t = NaN;
else
    t = tick(idx);
end
end

function s = local_source(replayMode)
if replayMode ~= 0
    s = 'replay';
else
    s = 'plant';
end
end

function parts = file_parts(pathChar, field)
[~, name, ~] = fileparts(pathChar);
switch field
    case 'name', parts = name;
    otherwise, parts = pathChar;
end
end
