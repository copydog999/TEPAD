function fault = tepadsim_make_fault(varargin)
%TEPADSIM_MAKE_FAULT Translates "the two injection buttons on the UI" into threshold constants the model can use.
%
%   fault = tepadsim_make_fault                          % use the defaults from tepadsim_config
%   fault = tepadsim_make_fault('spikeTick', 360, 'spikeSigma', 8)
%   fault = tepadsim_make_fault('shadeStart', 400, 'shadeDuration', 60, 'shadeFactor', 0.3)
%
%   Input semantics match inject_spike / inject_shading in Python model.py:
%     spikeTick     tick of the single-frame spike, amplitude = spikeSigma × the sigma at that tick; <0 = off
%     shadeStart    start of the shading interval (inclusive), length shadeDuration, half-open interval [start, start+duration)
%     shadeFactor   multiplicative attenuation inside the interval, (0,1); <=0 or >=1 means off
%
%   The output fields are the three thresholds referenced directly by Simulink Constant blocks:
%     spikeAt     matches tick==spikeAt; set to 1e9 when off (tick never equals it)
%     spikeSigma  amplitude multiplier
%     shadeFrom   tick>=shadeFrom; set to 1e9 when off
%     shadeTo     tick<shadeTo; set to -1e9 when off (together with shadeFrom this empties the interval)
%     shadeFactor attenuation factor
%     enabled     whether at least one channel is on
%   This way "off" needs no extra switch block: the comparators are naturally never true, as when no button is pressed on the MFC UI.

cfg = tepadsim_config();
spikeTick = cfg.fault.spikeTick;
spikeSigma = cfg.fault.spikeSigma;
shadeStart = cfg.fault.shadeStart;
shadeDuration = cfg.fault.shadeDuration;
shadeFactor = cfg.fault.shadeFactor;

for k = 1:2:numel(varargin)
    switch lower(varargin{k})
        case 'spiketick', spikeTick = varargin{k + 1};
        case 'spikesigma', spikeSigma = varargin{k + 1};
        case 'shadestart', shadeStart = varargin{k + 1};
        case 'shadeduration', shadeDuration = varargin{k + 1};
        case 'shadefactor', shadeFactor = varargin{k + 1};
        otherwise
            error('tepadsim:unknownOption', 'unknown option %s', varargin{k});
    end
end

fault.spikeSigma = double(spikeSigma);
fault.shadeFactor = double(shadeFactor);

if spikeTick >= 0
    fault.spikeAt = double(spikeTick);
else
    fault.spikeAt = 1e9;
end

shadeOn = shadeStart >= 0 && shadeDuration >= 1 && shadeFactor > 0 && shadeFactor < 1;
if shadeOn
    fault.shadeFrom = double(shadeStart);
    fault.shadeTo = double(shadeStart + shadeDuration);
else
    fault.shadeFrom = 1e9;
    fault.shadeTo = -1e9;
    fault.shadeFactor = 1.0;   % even an accidental match would not change the value
end

fault.enabled = (spikeTick >= 0) || shadeOn;
fault.raw = struct('spikeTick', spikeTick, 'spikeSigma', spikeSigma, ...
    'shadeStart', shadeStart, 'shadeDuration', shadeDuration, ...
    'shadeFactor', shadeFactor);
end
