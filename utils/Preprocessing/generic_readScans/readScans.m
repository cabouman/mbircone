function [ data ] = readScans( masterFile, plainParamsFile )

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../Modular_PreprocessingRoutines'));
addpath(fullfile(mfilepath,'../../misc_routines'));




disp(' ____________________________________________________')
disp('| Read File Names ...');
disp(' ----------------------------------------------------')
preprocessingParams = readPreprocessingParams(masterFile, plainParamsFile);
dataSetInfo = readDataSetInfo(masterFile, plainParamsFile)

printStruct(dataSetInfo, 'dataSetInfo');


disp(' ____________________________________________________')
disp('| Read Raw data ...');
disp(' ----------------------------------------------------')

viewIndexList = computeIndexList(preprocessingParams, dataSetInfo.par.N_beta, dataSetInfo.par.TotalAngle);


data.scans.object = read3D(dataSetInfo.scan, 'float32');
data.scans.object = data.scans.object(:,:,viewIndexList+1);

data.scans.driftReferenceStart_scan = zeros(size(data.scans.object(:,:,1)));
data.scans.driftReferenceEnd_scan = zeros(size(data.scans.object(:,:,1)));


data.scans.darkmeanImg = zeros(size(data.scans.object(:,:,1)));
data.scans.darkvarImg = zeros(size(data.scans.object(:,:,1)));


data.scans.blankmeanImg = read3D(dataSetInfo.blankScan, 'float32');
data.scans.blankvarImg = zeros(size(data.scans.object(:,:,1)));


data.scans

disp(' ____________________________________________________')
disp('| Compute Params ...');
disp(' ----------------------------------------------------')

data.par.u_s = dataSetInfo.par.u_s;
data.par.u_d0 = dataSetInfo.par.u_d0;
data.par.u_r = dataSetInfo.par.u_r;
data.par.v_r = dataSetInfo.par.v_r;
data.par.Delta_dv = dataSetInfo.par.Delta_dv;
data.par.Delta_dw = dataSetInfo.par.Delta_dw;
data.par.N_dv = dataSetInfo.par.N_dv;
data.par.N_dw = dataSetInfo.par.N_dw;
data.par.v_d0 = dataSetInfo.par.v_d0;
data.par.w_d0 = dataSetInfo.par.w_d0;
data.par.N_beta = length(viewIndexList);
data.par.TotalAngle = dataSetInfo.par.TotalAngle;
data.par.viewAngleList = computeAngleList( viewIndexList, dataSetInfo.par.N_beta, dataSetInfo.par.TotalAngle, preprocessingParams.rotationDirection)


disp(' ____________________________________________________')
disp('| Adjust Data ...');
disp(' ----------------------------------------------------')
%data.scans = flipScans(data.scans);



end

