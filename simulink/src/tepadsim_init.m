function varargout = tepadsim_init(varargin)
%TEPADSIM_INIT Prepares the three structs in the base workspace for the model (idempotent, never overwrites manual settings).
%
%   tepadsim_init                       % fill in whatever is missing
%   tepadsim_init('force', true)        % reset everything to defaults
%   s = tepadsim_init('scenario', struct('ticks',720,'seed',1301006,'replayMode',1), ...
%                     'fault', tepadsim_make_fault('spikeTick',360,'spikeSigma',8))
%
%   The three structs are the value source for every Constant/From Workspace block in the model:
%     tepadsimCalib     1x91 packed calibration + readable fields (tepadsim_load_calibration)
%     tepadsimScenario  plant model parameters, replay timeseries, tick count, replayMode
%     tepadsimFault     spike/shading threshold constants (tepadsim_make_fault)
%
%   The model's PreLoadFcn/InitFcn call it, so pressing Run directly in Simulink needs no script first;
%   the "fill in whatever is missing" behaviour keeps a tepadsimFault you tweaked in the command window from being wiped by InitFcn,
%   pass force=true explicitly to reset.

force = false;
silent = false;
scenarioSpec = struct();
faultSpec = struct();
calibFile = '';

for k = 1:2:numel(varargin)
    switch lower(varargin{k})
        case 'force', force = logical(varargin{k + 1});
        case 'silent', silent = logical(varargin{k + 1});
        case 'scenario', scenarioSpec = varargin{k + 1};
        case 'fault', faultSpec = varargin{k + 1};
        case 'calibfile', calibFile = varargin{k + 1};
        otherwise
            error('tepadsim:unknownOption', 'unknown option %s', varargin{k});
    end
end

cfg = tepadsim_setup_path();

% ---- calibration ---------------------------------------------------------
if force || ~local_has('tepadsimCalib')
    calib = tepadsim_load_calibration(calibFile);
    assignin('base', 'tepadsimCalib', calib);
    if ~silent
        fprintf('[tepadsim_init] calibration loaded: %s\n', calib.summary);
    end
end

% ---- scenario (data source) ----------------------------------------------
if force || ~local_has('tepadsimScenario')
    ticks = cfg.ticksPerDay;
    seed = cfg.seed;
    replayMode = 1;
    if isfield(scenarioSpec, 'ticks'), ticks = scenarioSpec.ticks; end
    if isfield(scenarioSpec, 'seed'), seed = scenarioSpec.seed; end
    if isfield(scenarioSpec, 'replayMode'), replayMode = scenarioSpec.replayMode; end
    scenario = tepadsim_generate_scenario('ticks', ticks, 'seed', seed);
    scenario.replayMode = replayMode;
    assignin('base', 'tepadsimScenario', scenario);
    if ~silent
        fprintf('[tepadsim_init] scenario generated: %d tick, seed=%d, data source=%s\n', ticks, seed, ...
            local_sourceName(replayMode));
    end
end

% ---- injection -----------------------------------------------------------
if force || ~local_has('tepadsimFault')
    if isfield(faultSpec, 'spikeTick') || isfield(faultSpec, 'shadeStart')
        opt = {};
        if isfield(faultSpec, 'spikeTick'), opt = [opt, {'spikeTick', faultSpec.spikeTick}]; end
        if isfield(faultSpec, 'spikeSigma')
            opt = [opt, {'spikeSigma', faultSpec.spikeSigma}];
        end
        if isfield(faultSpec, 'shadeStart')
            opt = [opt, {'shadeStart', faultSpec.shadeStart}];
        end
        if isfield(faultSpec, 'shadeDuration')
            opt = [opt, {'shadeDuration', faultSpec.shadeDuration}];
        end
        if isfield(faultSpec, 'shadeFactor')
            opt = [opt, {'shadeFactor', faultSpec.shadeFactor}];
        end
        fault = tepadsim_make_fault(opt{:});
    else
        fault = tepadsim_make_fault();
    end
    assignin('base', 'tepadsimFault', fault);
    if ~silent
        if fault.enabled
            fprintf('[tepadsim_init] injection: spikeAt=%g(×%gσ) shade=[%g,%g)×%g\n', ...
                fault.spikeAt, fault.spikeSigma, fault.shadeFrom, fault.shadeTo, fault.shadeFactor);
        else
            fprintf('[tepadsim_init] injection: off (baseline day)\n');
        end
    end
end

if nargout > 0
    varargout{1} = struct('calib', evalin('base', 'tepadsimCalib'), ...
        'scenario', evalin('base', 'tepadsimScenario'), ...
        'fault', evalin('base', 'tepadsimFault'));
end
end

function flag = local_has(name)
%Inside a callback the current workspace is base, but code inside a function must use evalin to see base variables.
flag = logical(evalin('base', sprintf('exist(''%s'', ''var'')', name)));
end

function s = local_sourceName(replayMode)
if replayMode ~= 0
    s = 'Replay Source (workspace replay)';
else
    s = 'PV Plant (in-model online generation)';
end
end
