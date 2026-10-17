function golden = tepadsim_load_golden(varargin)
%TEPADSIM_LOAD_GOLDEN Reads the per-tick ground-truth CSV exported by Python (sim_ui/selftest/golden).
%
%   golden = tepadsim_load_golden                 % default path
%   golden = tepadsim_load_golden(csvFile)        % explicit path
%
%   The file is produced by scripts/export_simulation_calibration.py --golden, header:
%     tick,G,P,point_z,short_mahalanobis,long_mahalanobis,score,alarm
%   It is the expected output for "the same calibration + the same simulation day" and the C++ self-test already uses it;
%   simulink treats it as the same source of truth so MATLAB does not get a second baseline.

cfg = tepadsim_setup_path();
if numel(varargin) >= 1 && ~isempty(varargin{1})
    file = varargin{1};
else
    file = cfg.goldenFile;
end
if exist(file, 'file') ~= 2
    error('tepadsim:missingGolden', ...
        ['golden file not found: %s\nrun this first: python scripts/export_simulation_calibration.py' ...
         ' --golden sim_ui/selftest/golden/golden_frames.csv'], file);
end

tbl = readtable(file, 'VariableNamingRule', 'preserve');
required = {'tick', 'G', 'P', 'point_z', 'short_mahalanobis', 'long_mahalanobis', 'score', 'alarm'};
for i = 1:numel(required)
    if ~ismember(required{i}, tbl.Properties.VariableNames)
        error('tepadsim:goldenHeader', 'golden file is missing column %s (actual: %s)', ...
            required{i}, strjoin(tbl.Properties.VariableNames, ', '));
    end
end

golden.tick = double(tbl.tick);
golden.g = double(tbl.G);
golden.p = double(tbl.P);
golden.pointZ = double(tbl.point_z);
golden.ms = double(tbl.short_mahalanobis);
golden.ml = double(tbl.long_mahalanobis);
golden.score = double(tbl.score);
golden.alarm = double(tbl.alarm) > 0.5;
golden.nTicks = height(tbl);
golden.file = file;
end
