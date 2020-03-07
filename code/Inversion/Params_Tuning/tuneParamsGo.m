function [] = tuneParamsGo( masterFile, plainParamsFile )

mfilepath=fileparts(which(mfilename));

addpath(fullfile(mfilepath,'MatlabRoutines'));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));
addpath(fullfile(mfilepath,'../Preprocessing/Modular_PreprocessingRoutines'));

binaryFNames = readBinaryFNames( masterFile, plainParamsFile );
sinoParams = readSinoParams(masterFile, plainParamsFile);

sino = read3D( binaryFNames.sino, 'float32');
errsino = read3D( binaryFNames.errsino, 'float32');
wght = read3D( binaryFNames.wght, 'float32');

% Stuff for generating file names 
[pathstr,name,ext] = fileparts(binaryFNames.sino);
subFolderName = '/Params_tuning/';
outFolder = [ pathstr, subFolderName ];
if(~(exist(outFolder, 'dir') == 7))
    mkdir(outFolder);
end

forwardProj = sino - errsino;

paramTuningParams.view_list = ':,:,1';

paramTuningParams.searchMethod.name = 'multiscaleSearch' ;
paramTuningParams.searchMethod.shift_searchRadius = 1;
paramTuningParams.searchMethod.shift_numPointsGrid = 10;
paramTuningParams.searchMethod.shift_gridReductionRatio = 5;
paramTuningParams.searchMethod.shift_gridSize = 0.001;

paramTuningParams.paramName = 'Delta_v_d0';
disp('Tuning Delta_v_d0...');
[Delta_v_d0_shift_voxel, error_list_all, s_list_all, error_scan] = tuneParam_single( paramTuningParams, 0, sino, forwardProj, wght );
Delta_v_d0_shift = Delta_v_d0_shift_voxel * sinoParams.Delta_dv ;
disp(['Delta_v_d0 <- Delta_v_d0 + ', num2str(Delta_v_d0_shift)]);

sList = s_list_all * sinoParams.Delta_dv;
eList = error_list_all;

str = '';
str = [str, 'sList = [', num2str(sList'), '];'];
str = [str, 'eList = [', num2str(eList'), '];'];
str = [str, 'plot(sList, eList, ''-o''); title(''Delta_{v_{d0}} error'')'];
str = [str, ''];
disp(str);

save('sList_eList', 'sList', 'eList')


fName_bestErr = [outFolder, name, '.bestErr', ext];
write3D( fName_bestErr, error_scan, 'float32');



