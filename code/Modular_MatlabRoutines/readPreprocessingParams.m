function [ preprocessingParams ] = readPreprocessingParams( masterFile, plainParamsFile )

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'preprocessingParams';
value = '';
resolveFlag = '';



% ############# DATA SELECTION            ###################################################
% ---------------- View selection -----------------------------------------------------------

preprocessingParams.subset_acquiredScans =              plainParams(executablePath, get_set, masterFile, masterField, 'subset_acquiredScans', value, resolveFlag);
preprocessingParams.N_beta_all =                str2num(plainParams(executablePath, get_set, masterFile, masterField, 'N_beta_all', value, resolveFlag));
preprocessingParams.num_timePoints =            str2num(plainParams(executablePath, get_set, masterFile, masterField, 'num_timePoints', value, resolveFlag));
preprocessingParams.index_timePoints =          str2num(plainParams(executablePath, get_set, masterFile, masterField, 'index_timePoints', value, resolveFlag));
preprocessingParams.num_viewSubsets =           str2num(plainParams(executablePath, get_set, masterFile, masterField, 'num_viewSubsets', value, resolveFlag));
preprocessingParams.index_viewSubsets =         str2num(plainParams(executablePath, get_set, masterFile, masterField, 'index_viewSubsets', value, resolveFlag));

preprocessingParams.rotationDirection =         str2num(plainParams(executablePath, get_set, masterFile, masterField, 'rotationDirection', value, resolveFlag));
preprocessingParams.N_avg =                     str2num(plainParams(executablePath, get_set, masterFile, masterField, 'N_avg', value, resolveFlag));
% ---------------- Cropping       -----------------------------------------------------------
preprocessingParams.crop_dv0 =                  str2num(plainParams(executablePath, get_set, masterFile, masterField, 'crop_dv0', value, resolveFlag));
preprocessingParams.crop_dv1 =                  str2num(plainParams(executablePath, get_set, masterFile, masterField, 'crop_dv1', value, resolveFlag));
preprocessingParams.crop_dw0 =                  str2num(plainParams(executablePath, get_set, masterFile, masterField, 'crop_dw0', value, resolveFlag));
preprocessingParams.crop_dw1 =                  str2num(plainParams(executablePath, get_set, masterFile, masterField, 'crop_dw1', value, resolveFlag));
% ---------------- Downscaling    -----------------------------------------------------------
preprocessingParams.downscale_dv =              str2num(plainParams(executablePath, get_set, masterFile, masterField, 'downscale_dv', value, resolveFlag));
preprocessingParams.downscale_dw =              str2num(plainParams(executablePath, get_set, masterFile, masterField, 'downscale_dw', value, resolveFlag));
% ---------------- ROR enlargemnt    --------------------------------------------------------
preprocessingParams.ROR_enlargeFactor_xy =      str2num(plainParams(executablePath, get_set, masterFile, masterField, 'ROR_enlargeFactor_xy', value, resolveFlag));


% ############# PARAMETER TUNING            #################################################
% ---------------- Geometry       -----------------------------------------------------------
preprocessingParams.Delta_u_s =                 str2num(plainParams(executablePath, get_set, masterFile, masterField, 'Delta_u_s', value, resolveFlag));
preprocessingParams.Delta_u_d0 =                str2num(plainParams(executablePath, get_set, masterFile, masterField, 'Delta_u_d0', value, resolveFlag));
preprocessingParams.Delta_v_d0 =                str2num(plainParams(executablePath, get_set, masterFile, masterField, 'Delta_v_d0', value, resolveFlag));
preprocessingParams.Delta_w_d0 =                str2num(plainParams(executablePath, get_set, masterFile, masterField, 'Delta_w_d0', value, resolveFlag));
preprocessingParams.Delta_axis_tilt =           str2num(plainParams(executablePath, get_set, masterFile, masterField, 'Delta_axis_tilt', value, resolveFlag));
preprocessingParams.tiltCorrectionMethod =              plainParams(executablePath, get_set, masterFile, masterField, 'tiltCorrectionMethod', value, resolveFlag);
preprocessingParams.Delta_v_r =                 str2num(plainParams(executablePath, get_set, masterFile, masterField, 'Delta_v_r', value, resolveFlag));
% ---------------- Voxel size     -----------------------------------------------------------
preprocessingParams.scaler_Delta_xy =           str2num(plainParams(executablePath, get_set, masterFile, masterField, 'scaler_Delta_xy', value, resolveFlag));
preprocessingParams.scaler_Delta_z =            str2num(plainParams(executablePath, get_set, masterFile, masterField, 'scaler_Delta_z', value, resolveFlag));


% ############# Sinogram Corrections        #################################################
% ---------------- Weight quantization  -----------------------------------------------------
preprocessingParams.weightCutoffPercentile =    str2num(plainParams(executablePath, get_set, masterFile, masterField, 'weightCutoffPercentile', value, resolveFlag));
% ---------------- BH Correction        -----------------------------------------------------
preprocessingParams.BHC_polynomial_coeffs =     str2num(plainParams(executablePath, get_set, masterFile, masterField, 'BHC_polynomial_coeffs', value, resolveFlag));
% ---------------- Source Shift         -----------------------------------------------------
preprocessingParams.shift_correctionType =      str2num(plainParams(executablePath, get_set, masterFile, masterField, 'shift_correctionType', value, resolveFlag));
preprocessingParams.shift_values =                      plainParams(executablePath, get_set, masterFile, masterField, 'shift_values', value, resolveFlag);
preprocessingParams.shift_searchRadius =        str2num(plainParams(executablePath, get_set, masterFile, masterField, 'shift_searchRadius', value, resolveFlag));
preprocessingParams.shift_gridSize =            str2num(plainParams(executablePath, get_set, masterFile, masterField, 'shift_gridSize', value, resolveFlag));
preprocessingParams.shift_numPointsGrid =       str2num(plainParams(executablePath, get_set, masterFile, masterField, 'shift_numPointsGrid', value, resolveFlag));
preprocessingParams.shift_gridReductionRatio =  str2num(plainParams(executablePath, get_set, masterFile, masterField, 'shift_gridReductionRatio', value, resolveFlag));
% ---------------- Sinogram Drift       -----------------------------------------------------
preprocessingParams.driftCorrection_type =      str2num(plainParams(executablePath, get_set, masterFile, masterField, 'driftCorrection_type', value, resolveFlag));
preprocessingParams.backgroundPatchLimits =             plainParams(executablePath, get_set, masterFile, masterField, 'backgroundPatchLimits', value, resolveFlag);
preprocessingParams.drift_spatialSigma =        str2num(plainParams(executablePath, get_set, masterFile, masterField, 'drift_spatialSigma', value, resolveFlag));
% ---------------- Occlution Correct       -----------------------------------------------------
preprocessingParams.occusionCorrectionType =    str2num(plainParams(executablePath, get_set, masterFile, masterField, 'occusionCorrectionType', value, resolveFlag));









end

