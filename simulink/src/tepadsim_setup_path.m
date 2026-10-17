function varargout = tepadsim_setup_path(varargin)
%TEPADSIM_SETUP_PATH Adds simulink/src to the MATLAB path and returns the configuration struct.
%
%   cfg = tepadsim_setup_path;  % add the path and return the tepadsim_config result
%
%   Model callbacks (PreLoadFcn/InitFcn) and every entry script call it first, so that no matter which
%   working directory MATLAB starts in, tepadsim_* can always be found.

cfg = tepadsim_config();
if ~ismember(cfg.srcDir, strsplit(path, pathsep))
    addpath(cfg.srcDir);
end

% The model/ directory is deliberately not added to the path: the model name would then match both the .slx on the path
% and the full path passed to load_system, and Simulink would emit a "model name is shadowing" warning on every load.
if exist(cfg.modelDir, 'dir') ~= 7
    mkdir(cfg.modelDir);
end

if nargout > 0
    varargout{1} = cfg;
end
end
