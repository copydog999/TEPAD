function delay = tepadsim_first_alarm_after(alarm, tick, startTick, horizon)
%TEPADSIM_FIRST_ALARM_AFTER Delay in ticks until the first alarm after injection; NaN if no alarm.
%
%   delay = tepadsim_first_alarm_after(alarm, tick, startTick, horizon)
%
%   The same quantity as the "shading injected → alarm 18 frames later" readout on the right of the sim_ui screen:
%   find the first tick in [startTick, startTick+horizon) where alarm is true and return its offset from startTick.
%   1 tick = 1 minute (config.SIMULATION_TICK_MINUTES), so the delay value is also a number of minutes.

alarm = alarm(:);
tick = tick(:);
stop = startTick + horizon;
hit = find(alarm & (tick >= startTick) & (tick < stop), 1);
if isempty(hit)
    delay = NaN;
else
    delay = tick(hit) - startTick;
end
end
