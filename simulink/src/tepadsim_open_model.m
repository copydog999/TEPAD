function tepadsim_open_model(varargin)
%TEPADSIM_OPEN_MODEL Opens the Simulink UI and prepares the workspace, for pressing Run by hand / changing the injection.
%
%   tepadsim_open_model                              % builds the model first if it does not exist
%   tepadsim_open_model('spikeTick', 360, 'spikeSigma', 8)   % open with one injection already applied
%   tepadsim_open_model('force', true)                % reset the workspace to a default normal day
%
%   Two ways to inject by hand in the UI (equivalent to the "inject spike / inject shading" buttons in sim_ui):
%     1) command line: tepadsim_open_model('spikeTick',360,'spikeSigma',8,'shadeStart',400,
%        'shadeDuration',60,'shadeFactor',0.3) and then press Run;
%     2) in the diagram: change the values of the Constant blocks Spike At / Spike Amp / Shade From / Shade To / Shade Factor
%        (they reference workspace variables, so you can also edit the fields of
%        tepadsimFault in the variables editor).
%
%   After a run, type tepadsimLog in the command window to see the 720x9 record, column order:
%     [tick g p pNormal score alarm pointZ ms ml]
%   Want figures directly: res = tepadsim_simulate(...) then tepadsim_plot_scan.
%
%   Options: force (default true, resets the workspace; pass false to only open the model and leave existing variables alone)
%   / spikeTick / spikeSigma / shadeStart / shadeDuration / shadeFactor
%   / ticks / seed; any other same-named options are passed through to tepadsim_build_model.

cfg = tepadsim_setup_path();
opt = struct('force', true, ...
    'ticks', cfg.ticksPerDay, 'seed', cfg.seed, ...
    'spikeTick', cfg.fault.spikeTick, 'spikeSigma', cfg.fault.spikeSigma, ...
    'shadeStart', cfg.fault.shadeStart, 'shadeDuration', cfg.fault.shadeDuration, ...
    'shadeFactor', cfg.fault.shadeFactor);
fn = fieldnames(opt);
extra = {};
for k = 1:2:numel(varargin)
    idx = find(strcmpi(fn, varargin{k}), 1);
    if isempty(idx)
        % Unknown options are treated as build_model's (e.g. 'openAfter')
        extra = [extra, {varargin{k}, varargin{k + 1}}]; %#ok<AGROW>
        continue;
    end
    opt.(fn{idx}) = varargin{k + 1};
end

fault = tepadsim_make_fault('spikeTick', opt.spikeTick, 'spikeSigma', opt.spikeSigma, ...
    'shadeStart', opt.shadeStart, 'shadeDuration', opt.shadeDuration, ...
    'shadeFactor', opt.shadeFactor);

% build + load + prepare base workspace (InitFcn also calls tepadsim_init; here it happens explicitly once)
tepadsim_build_model(extra{:});
if ~bdIsLoaded(cfg.modelName)
    load_system(cfg.modelFile);
end
tepadsim_init('force', opt.force, ...
    'scenario', struct('ticks', opt.ticks, 'seed', opt.seed, 'replayMode', 1), ...
    'fault', fault);
if usejava('desktop')
    open_system(cfg.modelName);
else
    % There is no desktop in -batch, so opening the diagram is pointless (it also spams warnings on exit); just prepare the variables.
    fprintf('[notice] running in -batch, no model window opened; workspace is ready.\n');
end

fprintf('model loaded: %s\n', cfg.modelFile);
raw = fault.raw;      % spikeAt/shadeFrom are "never matches when off" thresholds; the human-readable originals live in raw
if raw.spikeTick >= 0 || raw.shadeStart >= 0
    fprintf('  injection: spike %gσ @tick %s, shading ×%.2f/%d @tick %s\n', ...
        raw.spikeSigma, local_txt(raw.spikeTick), raw.shadeFactor, ...
        raw.shadeDuration, local_txt(raw.shadeStart));
else
    fprintf('  injection: off (baseline day). To inject, change the Constant blocks or rerun this function with parameters.\n');
end
fprintf('  press Run, then look at tepadsimLog in the command window (%dx9).\n', opt.ticks);
end

function s = local_txt(v)
if v < 0
    s = 'off';
else
    s = sprintf('%d', v);
end
end
