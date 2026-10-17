function ok = tepadsim_run_all(varargin)
%TEPADSIM_RUN_ALL Runs the whole Simulink pipeline with one call: self-test → build → checks → injection scan → plots.
%
%   ok = tepadsim_run_all
%   ok = tepadsim_run_all('rebuild', false, 'skipScan', true)
%
%   Options
%     rebuild   whether to rebuild the .slx (default true; pass false when the model exists and the wiring is unchanged)
%     skipScan  skip the injection scan and the plots (for looking at the check results only)
%     verbose   print each step's progress (default true)
%
%   Artifacts of the five steps:
%     1. tepadsim_selftest      pure MATLAB path vs Python golden (Simulink not started)
%     2. tepadsim_build_model   generates simulink/model/tepadsim_pv_tepad.slx
%     3. tepadsim_verify_model  three checks on the wired model → simulink/data/model_checks.csv
%     4. tepadsim_injection_scan spike/shading/mixed sweep → simulink/data/injection_scan.csv
%     5. tepadsim_plot_scan     two figures → simulink/figures/*.png|pdf
%
%   A failing step sets ok to false and the remaining diagnostics are still printed, so one run reveals every problem.

cfg = tepadsim_setup_path();
opt = struct('rebuild', true, 'skipScan', false, 'verbose', true);
fn = fieldnames(opt);
for k = 1:2:numel(varargin)
    idx = find(strcmpi(fn, varargin{k}), 1);
    if isempty(idx)
        error('tepadsim:unknownOption', 'unknown option %s', varargin{k});
    end
    opt.(fn{idx}) = varargin{k + 1};
end

ok = true;
local_banner(opt.verbose, 1, 'pure MATLAB path self-test (tepadsim_selftest)');
t0 = tic;
try
    pass1 = tepadsim_selftest('verbose', opt.verbose);
catch err
    pass1 = false;
    fprintf('  exception: %s\n', err.message);
end
ok = ok && pass1;

local_banner(opt.verbose, 2, 'building the Simulink model (tepadsim_build_model)');
try
    modelFile = tepadsim_build_model('force', opt.rebuild);
    if opt.verbose
        fprintf('  model: %s\n', modelFile);
    end
catch err
    ok = false;
    fprintf('  build failed: %s\n', err.message);
    return;
end

local_banner(opt.verbose, 3, 'three model checks (tepadsim_verify_model)');
try
    rep = tepadsim_verify_model('verbose', opt.verbose, ...
        'saveTable', fullfile(cfg.dataDir, 'model_checks.csv'));
    ok = ok && rep.pass;
catch err
    ok = false;
    fprintf('  check exception: %s\n', err.message);
end

if ~opt.skipScan
    local_banner(opt.verbose, 4, 'manual injection scan (tepadsim_injection_scan)');
    scan = [];
    try
        scan = tepadsim_injection_scan('verbose', opt.verbose);
    catch err
        ok = false;
        fprintf('  scan exception: %s\n', err.message);
    end

    local_banner(opt.verbose, 5, 'plotting (tepadsim_plot_scan)');
    if ~isempty(scan)
        try
            tepadsim_plot_scan('scan', scan, 'verbose', opt.verbose);
        catch err
            ok = false;
            fprintf('  plot exception: %s\n', err.message);
        end
    else
        fprintf('  scan produced nothing, skipping the plots\n');
    end
end

if opt.verbose
    fprintf('\ntotal %.1f s — %s\n', toc(t0), local_verdict(ok));
end
end

function local_banner(verbose, step, titleText)
if verbose
    fprintf('\n===== [%d] %s =====\n', step, titleText);
end
end

function s = local_verdict(ok)
if ok
    s = 'all passed';
else
    s = 'some steps failed (see the output above)';
end
end
