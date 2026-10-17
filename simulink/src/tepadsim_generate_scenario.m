function scenario = tepadsim_generate_scenario(varargin)
%TEPADSIM_GENERATE_SCENARIO Generates one simulation day tick by tick (the data source of the replay path).
%
%   scenario = tepadsim_generate_scenario;                       % use default parameters
%   scenario = tepadsim_generate_scenario('ticks', 720, 'seed', 1301006);
%   scenario = tepadsim_generate_scenario('spikeTick', 360, 'spikeSigma', 8, ...
%                                         'shadeStart', 500, 'shadeDuration', 60, ...
%                                         'shadeFactor', 0.3);
%
%   Internally it just calls tepadsim_plant_step repeatedly per tick (the same code as the
%   PV Plant MATLAB Function block in the .slx), so "in-model online generation" and "pre-generated
%   replay" should come out bit-identical — tepadsim_selftest checks exactly that.
%
%   Returned fields
%     scenario.tick / g / p / pNormal / sigma   column vectors of length ticks
%     scenario.replay    timeseries, 5 columns [g p pNormal sigma tick], for From Workspace
%     scenario.ticks     number of ticks (scalar)
%     scenario.plantParams  1x9 plant model parameters
%     scenario.fault     injection parameters (effective on the MATLAB side only; the model does it with the injection subsystem)

cfg = tepadsim_setup_path();

p = inputParser;
addParameter(p, 'ticks', cfg.ticksPerDay, @(x) x >= 100);
addParameter(p, 'seed', cfg.seed, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'temperatureBase', 25.0);
addParameter(p, 'noiseScale', 1.0);
addParameter(p, 'spikeTick', cfg.fault.spikeTick);
addParameter(p, 'spikeSigma', cfg.fault.spikeSigma);
addParameter(p, 'shadeStart', cfg.fault.shadeStart);
addParameter(p, 'shadeDuration', cfg.fault.shadeDuration);
addParameter(p, 'shadeFactor', cfg.fault.shadeFactor);
parse(p, varargin{:});

ticks = p.Results.ticks;
plantParams = [8.0 2.0 0.82 0.025 0.008 ticks p.Results.temperatureBase ...
    p.Results.noiseScale p.Results.seed];

% The LCG seed must fall in [1, 2^31-2]; 0 would lock the random stream at 0 forever.
rngSeed = mod(abs(plantParams(9)), 2147483646.0) + 1.0;
state = [0.0 1.0 rngSeed];   % [nSeen, prevPersistent, rngState]; the first frame skips the AR recursion

meas = zeros(ticks, 5);
for k = 1:ticks
    [row, state] = tepadsim_plant_step(plantParams, state);
    meas(k, :) = row;
end

scenario.tick = meas(:, 5);
scenario.g = meas(:, 1);
scenario.p = meas(:, 2);
scenario.pNormal = meas(:, 3);
scenario.sigma = meas(:, 4);
scenario.ticks = ticks;
scenario.seed = plantParams(9);        % original seed (the in-model plant parameters use it)
scenario.rngSeed = rngSeed;            % seed after folding into the LCG range
scenario.plantParams = plantParams;
scenario.plantState0 = [0.0 1.0 rngSeed];   % initial value of the plant model Unit Delay
scenario.replayMode = 1;                    % 1=replay (deterministic), 0=in-model online generation
scenario.fault = struct('spikeTick', p.Results.spikeTick, ...
    'spikeSigma', p.Results.spikeSigma, ...
    'shadeStart', p.Results.shadeStart, ...
    'shadeDuration', p.Results.shadeDuration, ...
    'shadeFactor', p.Results.shadeFactor);

% From Workspace consumes the timeseries directly: data matrix N x 5 = [g p pNormal sigma tick], time 0:1:N-1.
scenario.replay = timeseries(meas, (0:ticks-1).');
scenario.replay.Name = 'tepadsimReplay';
end
