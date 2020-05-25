function [ data ] = readScans_GE( masterFile, plainParamsFile )

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../Modular_PreprocessingRoutines'));
addpath(fullfile(mfilepath,'../../misc_routines'));



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Read File Names and such
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ____________________________________________________')
disp('| Read File Names ...');
disp(' ----------------------------------------------------')
preprocessingParams = readPreprocessingParams(masterFile, plainParamsFile);
dataSetInfo = readDataSetInfo(masterFile, plainParamsFile);

%masterPathNames = readMasterPathNames(masterFile);



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Read Files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ____________________________________________________')
disp('| Read Files ...');
disp(' ----------------------------------------------------')
[ data.scans, parPCA, viewIndexList ] = readAllPCAData( dataSetInfo, preprocessingParams );



disp(' ____________________________________________________')
disp('| Adjust Data ...');
disp(' ----------------------------------------------------')
data.scans = flipScans(data.scans);

% Computing geometry parameters
data.par = convertPCAparams2CBMBIRFormat( parPCA, preprocessingParams, viewIndexList);





end

