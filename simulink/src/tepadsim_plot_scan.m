function figs = tepadsim_plot_scan(varargin)
%TEPADSIM_PLOT_SCAN Plots the injection scan results as two figures (injection timeline + amplitude/detection comparison).
%
%   figs = tepadsim_plot_scan                       % run a scan first, then plot it
%   figs = tepadsim_plot_scan('scan', scan)         % reuse an existing scan struct
%   figs = tepadsim_plot_scan('outDir', cfg.figureDir, 'formats', {{'png','pdf'}})
%
%   Figure 1 tepadsim_fig_injection_timeline: the mixed-injection day (spike 8σ + shading ×0.30)
%       tick-by-tick timeline — irradiance/power, point residual |z|, normalized score and alarm markers;
%       a vertical dashed line marks the spike instant, a light band marks the shading interval.
%   Figure 2 tepadsim_fig_scan_delay: peak score and alarm count along the spike-amplitude and
%       shading-depth sweeps, matching the sim_ui screen's "how big an injection → how many alarms".
%
%   Both figures are drawn in invisible windows and exported by exportgraphics to png + vector pdf; undetected points are crossed out on the score plots.
%   Returns figs.png / figs.pdf, lists of absolute paths to the written files.

cfg = tepadsim_setup_path();
opt = struct('scan', [], ...
    'outDir', cfg.figureDir, ...
    'verbose', true, ...
    'formats', {{'png', 'pdf'}});
fn = fieldnames(opt);
for k = 1:2:numel(varargin)
    idx = find(strcmpi(fn, varargin{k}), 1);
    if isempty(idx)
        error('tepadsim:unknownOption', 'unknown option %s', varargin{k});
    end
    opt.(fn{idx}) = varargin{k + 1};
end

scan = opt.scan;
if isempty(scan)
    scan = tepadsim_injection_scan('verbose', opt.verbose, 'saveCsv', '');
end
if exist(opt.outDir, 'dir') ~= 7
    mkdir(opt.outDir);
end
calib = tepadsim_load_calibration();

fig1 = local_fig_timeline(scan, calib);
fig2 = local_fig_scan(scan);

figs = struct();
figs.png = {};
figs.pdf = {};
specs = {fig1, 'tepadsim_fig_injection_timeline'; fig2, 'tepadsim_fig_scan_delay'};
for i = 1:size(specs, 1)
    [pngList, pdfList] = local_save(specs{i, 1}, fullfile(opt.outDir, specs{i, 2}), opt.formats);
    figs.png = [figs.png, pngList];
    figs.pdf = [figs.pdf, pdfList];
    close(specs{i, 1});
end

if opt.verbose
    allFiles = [figs.png, figs.pdf];
    for i = 1:numel(allFiles)
        fprintf('  figure written: %s\n', allFiles{i});
    end
end
end

% ===========================================================================
%  Figure 1: injection timeline
% ===========================================================================

function fig = local_fig_timeline(scan, calib)
res = scan.demo;
tick = res.tick;
fault = res.fault;
t0 = min(tick);
t1 = max(tick);

fig = figure('Visible', 'off', 'Color', [1 1 1], 'Position', [40 40 980 680]);

% ---- top: irradiance and power -------------------------------------------
ax1 = subplot(3, 1, 1);
plot(ax1, tick, res.g, '-', 'Color', [0.20 0.45 0.80], 'LineWidth', 1.1, ...
    'DisplayName', 'irradiance G');
hold(ax1, 'on');
plot(ax1, tick, res.p, '-', 'Color', [0.85 0.55 0.15], 'LineWidth', 1.1, ...
    'DisplayName', 'power P (after injection)');
local_shade_band(ax1, scan.shadeStart, scan.shadeDuration, 'shading interval');
local_vline(ax1, scan.spikeTick, 'spike instant');
if res.nAlarm > 0
    local_alarm_marks(ax1, res);
end
xlim(ax1, [t0 t1]);
ylabel(ax1, 'G [W/m^2] / P [W]');
title(ax1, sprintf(['Simulation day timeline: spike %g\\sigma @ tick %d, shading \\times%.2f/%d tick @ tick %d' ...
    '(seed=%d)'], fault.spikeSigma, scan.spikeTick, fault.shadeFactor, ...
    scan.shadeDuration, scan.shadeStart, res.scenario.seed), 'FontSize', 10);
legend(ax1, 'Location', 'southwest', 'FontSize', 8);
grid(ax1, 'on');

% ---- middle: point residual |z| ------------------------------------------
ax2 = subplot(3, 1, 2);
plot(ax2, tick, abs(res.pointZ), '-', 'Color', [0.30 0.30 0.35], 'LineWidth', 1.0, ...
    'DisplayName', 'point residual |z|');
hold(ax2, 'on');
yline(ax2, calib.residualThreshold, '--', 'Color', [0.70 0.15 0.15], 'LineWidth', 1.0, ...
    'DisplayName', sprintf('\\tau_z = %.2f', calib.residualThreshold));
local_shade_band(ax2, scan.shadeStart, scan.shadeDuration, '');
local_vline(ax2, scan.spikeTick, '');
xlim(ax2, [t0 t1]);
ylim(ax2, [0 max([calib.residualThreshold * 1.2; abs(res.pointZ)])]);
ylabel(ax2, '|z|');
legend(ax2, 'Location', 'northwest', 'FontSize', 8);
grid(ax2, 'on');

% ---- bottom: normalized score and alarms ---------------------------------
ax3 = subplot(3, 1, 3);
plot(ax3, tick, res.score, '-', 'Color', [0.10 0.50 0.30], 'LineWidth', 1.1, ...
    'DisplayName', 'normalized score');
hold(ax3, 'on');
yline(ax3, 1.0, '--', 'Color', [0.70 0.15 0.15], 'LineWidth', 1.0, ...
    'DisplayName', 'threshold 1.0');
local_shade_band(ax3, scan.shadeStart, scan.shadeDuration, '');
local_vline(ax3, scan.spikeTick, '');
if res.nAlarm > 0
    alarmTick = res.tick(res.alarm);
    plot(ax3, alarmTick, res.score(res.alarm), 'rv', 'MarkerFaceColor', [0.75 0.15 0.15], ...
        'MarkerSize', 5, 'DisplayName', sprintf('%d alarms', res.nAlarm));
end
xlim(ax3, [t0 t1]);
xlabel(ax3, 'tick (1 tick = 1 minute)');
ylabel(ax3, 'score');
legend(ax3, 'Location', 'northwest', 'FontSize', 8);
grid(ax3, 'on');
end

% ===========================================================================
%  Figure 2: amplitude sweep
% ===========================================================================

function fig = local_fig_scan(scan)
spike = local_pick(scan.rows, 'spike');
shade = local_pick(scan.rows, 'shading');
baseScore = scan.base.maxScore;
xSpike = scan.spikeSigmas(:).';
xShade = scan.shadeFactors(:).';

fig = figure('Visible', 'off', 'Color', [1 1 1], 'Position', [40 40 980 640]);

% ---- spike: peak score ---------------------------------------------------
ax1 = subplot(2, 2, 1);
plot(ax1, xSpike, spike.maxScore, '-o', 'Color', [0.10 0.50 0.30], ...
    'MarkerFaceColor', [0.10 0.50 0.30], 'DisplayName', 'peak score');
hold(ax1, 'on');
yline(ax1, 1.0, '--', 'Color', [0.70 0.15 0.15], 'DisplayName', 'alarm threshold');
yline(ax1, baseScore, ':', 'Color', [0.45 0.45 0.50], 'DisplayName', 'baseline peak');
local_mark_undetected(ax1, xSpike, spike.maxScore, spike.detected);
xlabel(ax1, 'spike amplitude (\sigma)');
ylabel(ax1, 'peak score');
title(ax1, sprintf('spike @ tick %d: peak score', scan.spikeTick), 'FontSize', 10);
grid(ax1, 'on');
legend(ax1, 'Location', 'northwest', 'FontSize', 8);

% ---- spike: alarm count --------------------------------------------------
ax2 = subplot(2, 2, 2);
bar(ax2, xSpike, spike.nAlarm, 0.6, 'FaceColor', [0.20 0.45 0.80], ...
    'DisplayName', 'alarm count');
hold(ax2, 'on');
local_label_bars(ax2, xSpike, spike.nAlarm, spike.delay);
xlabel(ax2, 'spike amplitude (\sigma)');
ylabel(ax2, 'alarm count');
title(ax2, 'spike: alarm count (first delay annotated)', 'FontSize', 10);
grid(ax2, 'on');
ylabel(ax2, 'count');

% ---- shading: peak score -------------------------------------------------
ax3 = subplot(2, 2, 3);
plot(ax3, xShade, shade.maxScore, '-s', 'Color', [0.10 0.50 0.30], ...
    'MarkerFaceColor', [0.10 0.50 0.30], 'DisplayName', 'peak score');
hold(ax3, 'on');
yline(ax3, 1.0, '--', 'Color', [0.70 0.15 0.15], 'DisplayName', 'alarm threshold');
yline(ax3, baseScore, ':', 'Color', [0.45 0.45 0.50], 'DisplayName', 'baseline peak');
local_mark_undetected(ax3, xShade, shade.maxScore, shade.detected);
xlabel(ax3, 'shading factor (P \times)');
ylabel(ax3, 'peak score');
title(ax3, sprintf('shading @ tick %d / %d tick: peak score', ...
    scan.shadeStart, scan.shadeDuration), 'FontSize', 10);
set(ax3, 'XDir', 'reverse');
grid(ax3, 'on');
legend(ax3, 'Location', 'northwest', 'FontSize', 8);

% ---- shading: alarm count ------------------------------------------------
ax4 = subplot(2, 2, 4);
bar(ax4, xShade, shade.nAlarm, 0.6, 'FaceColor', [0.20 0.45 0.80], ...
    'DisplayName', 'alarm count');
hold(ax4, 'on');
local_label_bars(ax4, xShade, shade.nAlarm, shade.delay);
xlabel(ax4, 'shading factor (P \times)');
ylabel(ax4, 'alarm count');
title(ax4, 'shading: alarm count (first delay annotated)', 'FontSize', 10);
set(ax4, 'XDir', 'reverse');
grid(ax4, 'on');
end

% ===========================================================================
%  Shared helpers
% ===========================================================================

function s = local_pick(rows, kind)
% Pick the columns of interest out of scan.rows by kind, in the same order as the sweep grid.
s = struct('maxScore', [], 'nAlarm', [], 'detected', [], 'delay', []);
for r = 1:size(rows, 1)
    if strcmp(char(rows{r, 2}), kind)
        s.maxScore(end + 1) = rows{r, 8};
        s.nAlarm(end + 1) = rows{r, 7};
        s.detected(end + 1) = logical(rows{r, 6});
        s.delay(end + 1) = rows{r, 5};
    end
end
s.maxScore = s.maxScore(:).';
s.nAlarm = s.nAlarm(:).';
s.detected = s.detected(:).';
s.delay = s.delay(:).';
end

function local_mark_undetected(ax, x, y, detected)
% Mark undetected points with a cross: the score axis alone can suggest every case alarmed.
idx = ~detected;
if any(idx)
    plot(ax, x(idx), y(idx), 'kx', 'MarkerSize', 10, 'LineWidth', 1.5, ...
        'DisplayName', 'not detected');
end
end

function local_label_bars(ax, x, y, delay)
% Label the top of each bar with the first detection delay (tick); -1 is written as "not detected".
yl = ylim(ax);
ax.YLim(2) = max([yl(2); y(:)]) * 1.18;
yl = ylim(ax);
for i = 1:numel(x)
    if delay(i) < 0
        text(ax, x(i), y(i) + 0.04 * (yl(2) - yl(1)), 'not detected', ...
            'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', [0.6 0.1 0.1]);
    else
        text(ax, x(i), y(i) + 0.04 * (yl(2) - yl(1)), sprintf('delay %d', delay(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7);
    end
end
end

function local_vline(ax, x, label)
if isempty(x) || x < 0
    return;
end
xl = xlim(ax);
if x < xl(1) || x > xl(2)
    return;
end
if isempty(label)
    hd = 'off';
else
    hd = 'on';
end
plot(ax, [x x], ylim(ax), '--', 'Color', [0.70 0.15 0.15], 'LineWidth', 1.0, ...
    'DisplayName', label, 'HandleVisibility', hd);
end

function local_shade_band(ax, x0, dur, label)
if isempty(x0) || x0 < 0 || dur <= 0
    return;
end
a = x0;
b = x0 + dur;
yl = ylim(ax);
if isempty(label)
    hd = 'off';
else
    hd = 'on';
end
patch(ax, [a a b b], [yl(1) yl(2) yl(2) yl(1)], [0.85 0.40 0.20], ...
    'FaceAlpha', 0.10, 'EdgeColor', 'none', 'DisplayName', label, ...
    'HandleVisibility', hd);
end

function local_alarm_marks(ax, res)
% Alarm markers sit right on the power curve, easier to read than a band along the top.
idx = find(res.alarm);
if isempty(idx)
    return;
end
plot(ax, res.tick(idx), res.p(idx), 'v', 'Color', [0.75 0.15 0.15], ...
    'MarkerSize', 5, 'DisplayName', 'alarm instant');
end

function [pngList, pdfList] = local_save(fig, baseName, formats)
pngList = {};
pdfList = {};
for i = 1:numel(formats)
    fmt = char(formats{i});
    file = fullfile([baseName '.' fmt]);
    try
        if strcmp(fmt, 'png')
            exportgraphics(fig, file, 'Resolution', 200);
        else
            exportgraphics(fig, file, 'ContentType', 'vector');
        end
    catch err
        if strcmp(fmt, 'png')
            print(fig, '-dpng', '-r200', file);
        else
            print(fig, '-dpdf', file);
        end
        warning('tepadsim:exportFallback', '%s fell back to print: %s', file, err.message);
    end
    if strcmp(fmt, 'png')
        pngList = [pngList, {file}];
    else
        pdfList = [pdfList, {file}];
    end
end
end
