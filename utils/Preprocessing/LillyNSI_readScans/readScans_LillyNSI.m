function [ data ] = readScans_LillyNSI(masterFile, plainParamsFile )

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'MatlabRoutines'));
addpath(fullfile(mfilepath,'../Modular_PreprocessingRoutines'));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));
addpath(fullfile(mfilepath,'../../../Modular_MatlabRoutines'));



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Read File Names
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Read File Names ...');
preprocessingParams = readPreprocessingParams(masterFile, plainParamsFile);

dataSetInfo = readDataSetPath_LillyNSI(masterFile, plainParamsFile)
sysParams = getSysParams(dataSetInfo)

indexList = computeIndexList(preprocessingParams, sysParams.numAcquiredScans, sysParams.TotalAngle);


disp(' ____________________________________________________')
disp('| Read Raw data ...');
disp(' ----------------------------------------------------')

% Read Scans 
data.scans = read_scan_data(dataSetInfo, indexList)
disp('Scan sizes:')
disp(size(data.scans.object))


data.scans = mirror_scans( data.scans, 'vert' );
% Mirror scans
if dataSetInfo.do_mirrorScans_Horz==1
	data.scans = mirror_scans( data.scans, 'horz' );
end
if dataSetInfo.do_mirrorScans_Vert==1
	data.scans = mirror_scans( data.scans, 'vert' );
end

% Rotate Scans
if dataSetInfo.timesRotateScans90~=0
	data.scans = rotate_scans( data.scans, dataSetInfo.timesRotateScans90 );
end


disp(' ____________________________________________________')
disp('| Compute Params ...');
disp(' ----------------------------------------------------')

% Get params
data.par = convert_sysParams_to_MBIRformat_manual(sysParams, indexList, preprocessingParams, dataSetInfo, data.scans);


% Crop scans
cropLen = dataSetInfo.cropLen;
data.scans.object = cropScan(data.scans.object, cropLen, cropLen, cropLen, cropLen);
data.scans.darkmeanImg = cropScan(data.scans.darkmeanImg, cropLen, cropLen, cropLen, cropLen);
data.scans.blankmeanImg = cropScan(data.scans.blankmeanImg, cropLen, cropLen, cropLen, cropLen);
data.scans.darkvarImg = cropScan(data.scans.darkvarImg, cropLen, cropLen, cropLen, cropLen);
data.scans.blankvarImg = cropScan(data.scans.blankvarImg, cropLen, cropLen, cropLen, cropLen);
data.scans.driftReferenceStart_scan = cropScan(data.scans.driftReferenceStart_scan, cropLen, cropLen, cropLen, cropLen);
data.scans.driftReferenceEnd_scan = cropScan(data.scans.driftReferenceEnd_scan, cropLen, cropLen, cropLen, cropLen);
data.scans.defectivePixelMap = cropScan(data.scans.defectivePixelMap, cropLen, cropLen, cropLen, cropLen);
data.scans.jig_scan = cropScan(data.scans.jig_scan, cropLen, cropLen, cropLen, cropLen);

disp('Cropped Sizes');
disp(size(data.scans.object))
disp(size(data.scans.jig_scan))


% Adjust for cropped scans
data.par.w_d0 = data.par.w_d0 + round( (data.par.N_dw - size(data.scans.object,1)) * data.par.Delta_dw /2 );
data.par.v_d0 = data.par.v_d0 + round( (data.par.N_dv - size(data.scans.object,2)) * data.par.Delta_dv /2 );
data.par.N_dw = size(data.scans.object,1);
data.par.N_dv = size(data.scans.object,2);

fprintf('_____________________________________________________________________\n');
fprintf('scans.blankmeanImg_cropped(3,3): %s\n', data.scans.blankmeanImg(3,3));
fprintf('_____________________________________________________________________\n');

return


