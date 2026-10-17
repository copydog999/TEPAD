function cfg = tepadsim_config()
%TEPADSIM_CONFIG Shared configuration for the simulink subsystem (paths + constants aligned with src/config.py).
%
%   cfg = tepadsim_config; returns a struct; field meanings are in the inline comments below.
%   This deliberately depends only on mfilename('fullpath'), not on the current working directory,
%   so it works from the repo root, simulink/ or any MATLAB command-window location.

thisFile = mfilename('fullpath');
if isempty(thisFile)
    error('tepadsim:badInvocation', 'tepadsim_config must be called as a file');
end
srcDir = fileparts(thisFile);
root = fileparts(srcDir);              % simulink/
projectRoot = fileparts(root);         % TEPAD/

cfg.root = root;
cfg.srcDir = srcDir;
cfg.modelDir = fullfile(root, 'model');
cfg.dataDir = fullfile(root, 'data');
cfg.figureDir = fullfile(root, 'figures');
cfg.projectRoot = projectRoot;

cfg.modelName = 'tepadsim_pv_tepad';
cfg.modelFile = fullfile(cfg.modelDir, [cfg.modelName '.slx']);

% Calibration and golden both reuse the artifacts already produced on the Python/C++ side, avoiding a second source of truth
cfg.calibrationFile = fullfile(projectRoot, 'sim_ui', 'calibration', 'dual_window_calibration.json');
cfg.goldenFile = fullfile(projectRoot, 'sim_ui', 'selftest', 'golden', 'golden_frames.csv');
cfg.calibrationModelFile = fullfile(cfg.dataDir, 'dual_window_calibration.json');

% ---- simulation constants (one-to-one with SIMULATION_* / STREAM_* in src/config.py) --------
cfg.tickMinutes = 1;          % SIMULATION_TICK_MINUTES: 1 simulation step = 1 tick = 1 minute
cfg.ticksPerDay = 720;        % SIMULATION_DAY_TICKS
cfg.shortWindow = 10;         % STREAM_SHORT_WINDOW
cfg.longWindow = 60;          % STREAM_LONG_WINDOW
cfg.ewmaAlpha = 0.1;          % STREAM_EWMA_ALPHA (causal EWMA coefficient inside the detector)
cfg.seed = 601006 + 700000;   % SIMULATION_SEED + 700000, offset from the golden day's calibration seed
cfg.dayPhase = pi;            % 12-hour daylight half period

% ---- PV plant model parameters: aligned with model.py::TrueParameters + simulate_normal_day --
% 1x9 [alpha beta eta sigma0 sigma_min nTicks temperatureBase noiseScale seed]
% (the temperature coefficient pvGamma=-0.004 lives in tepadsim_plant_step, same as the model.py default,
%   which is why it is not in the vector.)
cfg.plantParams = [8.0 2.0 0.82 0.025 0.008 cfg.ticksPerDay 25.0 1.0 cfg.seed];

% ---- fault injection defaults: aligned with the default magnitudes of the two "manual injection" buttons in the sim_ui UI ----------
% The spike multiplier is relative to residual_sigma (MFC UI default 8σ); shading ×0.30 lasts 60 tick.
cfg.fault.spikeTick = -1;     % <0 = off (tick is always >=0, so the comparison never holds)
cfg.fault.spikeSigma = 8.0;   % spike amplitude = spikeSigma × sigma of that tick
cfg.fault.shadeStart = -1;    % <0 = off
cfg.fault.shadeDuration = 60;
cfg.fault.shadeFactor = 0.30;

% Detector state vector size: 1(nSeen) + 2*short + 2*long + 2*5(theta) + 2(ready)
cfg.stateDim = 1 + 2 * cfg.shortWindow + 2 * cfg.longWindow + 2 * 5 + 2;
% Packed calibration vector size: 11 scalars + 2 × (3*5 statistics + 25 inverse covariance)
cfg.calibDim = 11 + 2 * (15 + 25);
end
