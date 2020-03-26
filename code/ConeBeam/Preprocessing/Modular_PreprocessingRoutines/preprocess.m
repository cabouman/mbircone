function [ ] = preprocess(data, masterFile, plainParamsFile )

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(genpath(fullfile(mfilepath,'../Modular_PreprocessingRoutines')));
addpath(genpath(fullfile(mfilepath,'../../Modular_MatlabRoutines')));
addpath(genpath(fullfile(mfilepath,'../../../Modular_MatlabRoutines')));


disp(' ----------------------------------------------------')
disp('| Read File Names ...');
disp(' ----------------------------------------------------')
preprocessingParams = readPreprocessingParams(masterFile, plainParamsFile);
binaryFNames = readBinaryFNames( masterFile, plainParamsFile );


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Adjust Sinogram parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[data.par] = adjustParameters(data.par, preprocessingParams);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Software Patch for modules that may not be used
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(~isfield(data.scans, 'jig_scan'))
	data.scans.jig_scan = zeros(size(data.scans.object(:,:,1)));
end

if(~isfield(data.scans, 'defectivePixelMap'))
	data.scans.defectivePixelMap = zeros(size(data.scans.object(:,:,1)));
end


disp(' ----------------------------------------------------')
disp('| Downscaling ...');
disp(' ----------------------------------------------------')
[ data.scans, data.par ] = downscaleScansAndParams( data.scans, data.par , preprocessingParams);

disp(' ----------------------------------------------------')
disp('| Cropping ...');
disp(' ----------------------------------------------------')
[ data.scans, data.par ] = cropScansAndParams( data.scans, data.par, preprocessingParams );

checkCenter(data.par);


disp(' ----------------------------------------------------')
disp('| Computing sinogram ...');
disp(' ----------------------------------------------------')
[sino, driftReference_sino, jigMeasurementsSino, wght] = computeSinoAndWeight( data.scans, data.par, preprocessingParams);

[sino, wght, driftReference_sino, jigMeasurementsSino] = correct_defectivePixels( sino, wght, driftReference_sino, jigMeasurementsSino, data.scans.defectivePixelMap, 'interpolate' );


disp(' ----------------------------------------------------')
disp('| Correcting for shift in X-Ray focus ...')
disp(' ----------------------------------------------------')
[sino, wght, driftReference_sino, jigMeasurementsSino] = correct_shift( sino, wght, driftReference_sino, jigMeasurementsSino, data.par.TotalAngle, data.par.viewAngleList, preprocessingParams);


disp(' ----------------------------------------------------')
disp('| Correcting for drift in X-Ray photoncount ...')
disp(' ----------------------------------------------------')
[sino, wght, driftSino] = correct_drift( sino, wght, driftReference_sino, data.par.TotalAngle, data.par.viewAngleList, preprocessingParams);


disp(' ----------------------------------------------------')
disp('| Correcting for tilt in rotation axis ...')
disp(' ----------------------------------------------------')
[sino, wght, jigMeasurementsSino] = rotateSinoAndWeight( sino, wght, jigMeasurementsSino, preprocessingParams);
[data.par] = adjustParameters_axisTilt(data.par, preprocessingParams);


disp(' ----------------------------------------------------')
disp('| Apply BHC Polynomial ...');
disp(' ----------------------------------------------------')
origSino = sino;
gam = preprocessingParams.BHC_polynomial_coeffs;
disp(gam)
if(~isequal(gam, [0 1]))
	sino = polynomial_gam(origSino, gam);
end


disp(' ----------------------------------------------------')
disp('| Write Data to file ...')
disp(' ----------------------------------------------------')
write3D( binaryFNames.sino, sino, 'float32');
write3D( binaryFNames.origSino, origSino, 'float32');
write3D( binaryFNames.driftSino, driftSino, 'float32');
write3D( binaryFNames.jigMeasurementsSino, jigMeasurementsSino, 'float32');
write3D( binaryFNames.wght, wght, 'float32');


disp(' ----------------------------------------------------')
disp('| Write Params to file ...')
disp(' ----------------------------------------------------')
data.parImg = computeImgParams( data.par, preprocessingParams );
checkImageParams(data);
sinoParams = data.par;
imageParams = data.parImg;
viewAngleList = data.par.viewAngleList;

writeSinoParamsFile( sinoParams, masterFile, plainParamsFile );
writeImgParamsFile( imageParams, masterFile, plainParamsFile );
writeViewAngleList( viewAngleList, masterFile, plainParamsFile );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Display Params
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
preprocessingParams
sinoParams
imageParams

end






