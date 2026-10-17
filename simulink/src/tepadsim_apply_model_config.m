function tepadsim_apply_model_config(varargin)
%TEPADSIM_APPLY_MODEL_CONFIG Synchronizes the workspace scenario length to the model's StopTime.
%
%   tepadsim_apply_model_config          % applies to the current model (this is how InitFcn calls it)
%   tepadsim_apply_model_config(modelName)
%
%   The model's StopTime holds a literal (it must evaluate at load time and cannot depend on workspace variables),
%   so after changing ticks this function syncs it once from InitFcn,
%   while the command-line path relies on tepadsim_simulate setting it explicitly.

if nargin >= 1 && ~isempty(varargin{1})
    modelName = varargin{1};
else
    modelName = local_current_model();
end

scenario = evalin('base', 'tepadsimScenario');
set_param(modelName, 'StopTime', num2str(scenario.ticks - 1));
end

function name = local_current_model()
try
    name = bdroot(gcs);
catch
    cfg = tepadsim_config();
    name = cfg.modelName;
end
if isempty(name)
    cfg = tepadsim_config();
    name = cfg.modelName;
end
end
