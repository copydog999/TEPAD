function calib = tepadsim_load_calibration(varargin)
%TEPADSIM_LOAD_CALIBRATION Reads the sim_ui dual-window calibration JSON and packs it into the 1x91 vector the detector needs.
%
%   calib = tepadsim_load_calibration                 % default path
%   calib = tepadsim_load_calibration(jsonPath)       % explicit path
%
%   The data source is exactly the same as the C++ host application (exported by scripts/export_simulation_calibration.py);
%   nothing is re-fitted here, it only moves "JSON → fixed-length vector", and the field order is the packing
%   layout documented in the header comment of tepadsim_detector_step.m.
%
%   Returned struct
%     calib.packed          1x91 constant vector, wired straight into the calib input of the MATLAB Function
%     calib.etaReference    the scalars below are for printing/plotting
%     calib.residualSigma / calib.residualThreshold / calib.tauShort / calib.tauLong
%     calib.shortWindow / calib.longWindow / calib.ewmaAlpha / calib.zClip
%     calib.summary         one-line summary (same purpose as the C++ Calibration.summary)

cfg = tepadsim_setup_path();
if numel(varargin) >= 1 && ~isempty(varargin{1})
    jsonFile = varargin{1};
else
    jsonFile = cfg.calibrationFile;
end
if exist(jsonFile, 'file') ~= 2
    error('tepadsim:missingCalibration', ...
        ['calibration file not found: %s\nrun this first: python scripts/export_simulation_calibration.py' ...
         ' --golden sim_ui/selftest/golden/golden_frames.csv'], jsonFile);
end

raw = jsondecode(fileread(jsonFile));

% --- detector structure parameters: the window lengths must match the state layout (10/60), otherwise fail right here ---
shortWindow = 10;
longWindow = 60;
ewmaAlpha = 0.1;
zClip = 12.0;
sigmaFloorRate = 0.025;
gFloor = 0.1;
if isfield(raw, 'detector')
    d = raw.detector;
    if isfield(d, 'short_window') && d.short_window ~= shortWindow
        error('tepadsim:windowMismatch', 'calibration short_window=%d does not match the model state layout %d', ...
            d.short_window, shortWindow);
    end
    if isfield(d, 'long_window') && d.long_window ~= longWindow
        error('tepadsim:windowMismatch', 'calibration long_window=%d does not match the model state layout %d', ...
            d.long_window, longWindow);
    end
    if isfield(d, 'ewma_alpha'), ewmaAlpha = d.ewma_alpha; end
    if isfield(d, 'z_clip'), zClip = d.z_clip; end
    if isfield(d, 'sigma_floor_rate'), sigmaFloorRate = d.sigma_floor_rate; end
    if isfield(d, 'g_floor'), gFloor = d.g_floor; end
end

shortRef = local_check_reference(raw.short_reference, 'short_reference', 5);
longRef = local_check_reference(raw.long_reference, 'long_reference', 5);

calib.etaReference = double(raw.eta_reference);
calib.residualSigma = double(raw.residual_sigma);
calib.residualThreshold = double(raw.residual_threshold);
calib.tauShort = double(raw.tau_short);
calib.tauLong = double(raw.tau_long);
calib.shortWindow = shortWindow;
calib.longWindow = longWindow;
calib.ewmaAlpha = ewmaAlpha;
calib.zClip = zClip;
calib.sigmaFloorRate = sigmaFloorRate;
calib.gFloor = gFloor;
calib.shortReference = shortRef;
calib.longReference = longRef;
calib.sourceFile = jsonFile;

% --- packing (flatten the 5x5 inverse covariance column-major; the unpack side restores it with reshape(.,5,5)) ---
calib.packed = [ ...
    calib.etaReference, calib.residualSigma, calib.residualThreshold, ...
    calib.tauShort, calib.tauLong, calib.zClip, calib.sigmaFloorRate, calib.gFloor, ...
    shortWindow, longWindow, ewmaAlpha, ...
    shortRef.center, shortRef.scale, shortRef.standardizedCenter, shortRef.inverseCovariance(:).', ...
    longRef.center, longRef.scale, longRef.standardizedCenter, longRef.inverseCovariance(:).' ...
    ];
if numel(calib.packed) ~= 91
    error('tepadsim:packError', 'packed calibration vector length %d != 91', numel(calib.packed));
end

calib.summary = sprintf(['eta=%.6f z_sigma=%.6f tau_z=%.4f tau_short=%.4f tau_long=%.4f ' ...
    '(L=%d/%d, seed=%d)'], calib.etaReference, calib.residualSigma, ...
    calib.residualThreshold, calib.tauShort, calib.tauLong, shortWindow, longWindow, ...
    local_seed(raw));
end

function s = local_seed(raw)
if isfield(raw, 'calibration_seed')
    s = raw.calibration_seed;
else
    s = NaN;
end
end

function ref = local_check_reference(node, name, dim)
%Validate the dimensions field by field: better to fail here than let the detector silently compute nonsense.
ref.center = local_vector(node.center, dim, [name '.center']);
ref.scale = local_vector(node.scale, dim, [name '.scale']);
ref.standardizedCenter = local_vector(node.standardized_center, dim, [name '.standardized_center']);
cov = double(node.inverse_covariance);
if size(cov, 1) ~= dim || size(cov, 2) ~= dim
    error('tepadsim:badReference', '%s.inverse_covariance must be %dx%d', name, dim, dim);
end
ref.inverseCovariance = cov;
if isfield(node, 'threshold')
    ref.threshold = double(node.threshold);
end
if isfield(node, 'n_reference')
    ref.nReference = double(node.n_reference);
end
end

function v = local_vector(node, dim, label)
v = double(node(:).');
if numel(v) ~= dim
    error('tepadsim:badReference', '%s must have length %d, got %d', label, dim, numel(v));
end
end
