function res = tepadsim_reference_run(g, p, calib, varargin)
%TEPADSIM_REFERENCE_RUN Runs the detector tick by tick in pure MATLAB (without the Simulink solver).
%
%   res = tepadsim_reference_run(g, p, calib)
%   res = tepadsim_reference_run(g, p, calib, 'stateInit', s0)
%
%   It calls the same tepadsim_detector_step.m as the MATLAB Function block in the .slx,
%   so it acts as the "offline shadow of the model":
%     * the golden check uses it to prove the algorithm itself matches Python;
%     * after a model run it is used once more to prove the Simulink wiring/sampling timing adds no deviation.
%
%   The fields of res are all column vectors of length = numel(g):
%     tick g p score alarm pointZ ms ml shortReady longReady stateEnd

cfg = tepadsim_setup_path();
state = zeros(1, cfg.stateDim);
for k = 1:2:numel(varargin)          % supports 'stateInit', s0
    if strcmpi(varargin{k}, 'stateInit')
        state = varargin{k + 1}(:).';
    end
end

n = numel(g);
g = g(:);
p = p(:);

out = zeros(n, 7);
for k = 1:n
    [row, state] = tepadsim_detector_step(g(k), p(k), calib.packed(:).', state);
    out(k, :) = row;
end

res.tick = (0:n-1).';
res.g = g;
res.p = p;
res.score = out(:, 1);
res.alarm = out(:, 2) > 0.5;
res.pointZ = out(:, 3);
res.ms = out(:, 4);
res.ml = out(:, 5);
res.shortReady = out(:, 6) > 0.5;
res.longReady = out(:, 7) > 0.5;
res.stateEnd = state;
res.nAlarm = sum(res.alarm);
res.firstAlarmTick = local_first_true(res.alarm);
end

function t = local_first_true(logicalVec)
idx = find(logicalVec, 1);
if isempty(idx)
    t = NaN;
else
    t = idx - 1;   % ticks are counted from 0
end
end
