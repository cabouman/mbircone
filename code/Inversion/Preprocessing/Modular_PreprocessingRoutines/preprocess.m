function [ ] = preprocess(data, masterFile, plainParamsFile )

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(genpath(fullfile(mfilepath,'../Modular_PreprocessingRoutines')));
addpath(genpath(fullfile(mfilepath,'../../../Modular_MatlabRoutines')));


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Read File Names and such
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ____________________________________________________')
disp('| Read File Names ...');
disp(' ----------------------------------------------------')
preprocessingParams = readPreprocessingParams(masterFile, plainParamsFile);
binaryFNames = readBinaryFNames( masterFile, plainParamsFile );


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Adjust Sinogram parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[data.par] = adjustParameters(data.par, preprocessingParams);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Patches for modules that may not be used
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(~isfield(data.scans, 'occlusion_scan'))
	data.scans.occlusion_scan = zeros(size(data.scans.object(:,:,1)));
end

if(~isfield(data.scans, 'defectivePixelMap'))
	data.scans.defectivePixelMap = zeros(size(data.scans.object(:,:,1)));
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Downscaling and cropping
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ____________________________________________________')
disp('| Downscaling ...');
disp(' ----------------------------------------------------')
% Downscaling
[ data.scans, data.par ] = downscaleScansAndParams( data.scans, data.par , preprocessingParams);

disp(' ____________________________________________________')
disp('| Cropping ...');
disp(' ----------------------------------------------------')
% cropping
[ data.scans, data.par ] = cropScansAndParams( data.scans, data.par, preprocessingParams );

checkCenter(data.par);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Compute Sinogram
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ____________________________________________________')
disp('| Computing sinogram ...');
disp(' ----------------------------------------------------')
% Compute Sinogram and weight
[sino, driftReference_sino, occlusion_sino, wght] = computeSinoAndWeight( data.scans, data.par, preprocessingParams);

% Adjust for defective pixels
[sino, wght, driftReference_sino, occlusion_sino] = correct_defectivePixels( sino, wght, driftReference_sino, occlusion_sino, data.scans.defectivePixelMap, 'interpolate' );

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Correct for shifts in X-ray focal point
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ____________________________________________________')
disp('| Correcting for shift in X-Ray focus ...')
disp(' ----------------------------------------------------')
[sino, wght, driftReference_sino, occlusion_sino] = correct_shift( sino, wght, driftReference_sino, occlusion_sino, data.par.TotalAngle, data.par.viewAngleList, preprocessingParams);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Correct for drifts in Sinogram
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ____________________________________________________')
disp('| Correcting for drift in X-Ray photoncount ...')
disp(' ----------------------------------------------------')
[sino, wght, driftSino] = correct_drift( sino, wght, driftReference_sino, data.par.TotalAngle, data.par.viewAngleList, preprocessingParams);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Adjust For axis tilt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ____________________________________________________')
disp('| Correcting for tilt in rotation axis ...')
disp(' ----------------------------------------------------')
[sino, wght, occlusion_sino] = rotateSinoAndWeight( sino, wght, occlusion_sino, preprocessingParams);
[data.par] = adjustParameters_axisTilt(data.par, preprocessingParams);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Apply BHC Polynomial
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ____________________________________________________')
disp('| Apply BHC Polynomial ...');
disp(' ----------------------------------------------------')
origSino = sino;
gam = preprocessingParams.BHC_polynomial_coeffs;
if(~isequal(gam, [0 1]))
	sino = polynomial_gam(origSino, gam);
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Storing 3D data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ____________________________________________________')
disp('| Write Data to file ...')
disp(' ----------------------------------------------------')
write3D( binaryFNames.sino, sino, 'float32');
write3D( binaryFNames.driftSino, driftSino, 'float32');
write3D( binaryFNames.origSino, origSino, 'float32');
% WEIGHTCHANGE
write3D( binaryFNames.wght, wght, 'float32');
% wght = round( 255 * wght / prctile(wght(:),preprocessingParams.weightCutoffPercentile) );
% write3D( binaryFNames.wght, wght, 'uint8');



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Storing params data and view angles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Image parameters 
data.parImg = computeImgParams( data.par, preprocessingParams );
checkImageParams(data);


% Write param files file
writeSinoParamsFile( data.par, masterFile, plainParamsFile );
writeImgParamsFile( data.parImg, masterFile, plainParamsFile );

% Write angle list file
writeViewAngleList( data.par.viewAngleList, masterFile, plainParamsFile );

%%
preprocessingParams
sinoParams = data.par
imageParams = data.parImg


end






