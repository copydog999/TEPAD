function p = tepadsim_apply_fault(pIn, sigma, tick, fault)
%TEPADSIM_APPLY_FAULT MATLAB-side fault injection, equivalent to the injection subsystem in the model.
%
%   p = tepadsim_apply_fault(pIn, sigma, tick, fault)
%
%   One-to-one with three functions in Python model.py:
%     * inject_spike(power, tick, amplitude): amplitude = spikeSigma * sigma(tick),
%       affecting only 1 frame (this is the "inject spike (×σ)" button in the sim_ui UI);
%     * inject_shading(power, start, duration, factor): inside the half-open interval
%       [start, start+duration) multiply by factor;
%     * applying both together is inject_mixed.
%
%   fault fields: spikeTick / spikeSigma / shadeStart / shadeDuration / shadeFactor.
%   spikeTick < 0 or shadeStart < 0 disables that channel — exactly the same way
%   the Simulink model disables it ("Compare To Constant never holds").

p = pIn(:);
tick = tick(:);
sigma = sigma(:);
n = numel(p);

spikeTick = fault.spikeTick;
if spikeTick >= 0
    idx = find(tick == spikeTick, 1);
    if ~isempty(idx) && idx <= n
        p(idx) = p(idx) + fault.spikeSigma * sigma(idx);
    end
end

shadeStart = fault.shadeStart;
if shadeStart >= 0
    stop = shadeStart + fault.shadeDuration;          % half-open interval [start, stop)
    mask = (tick >= shadeStart) & (tick < stop);
    p(mask) = p(mask) * fault.shadeFactor;
end
end
