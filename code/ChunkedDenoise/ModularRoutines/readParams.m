function [ par ] = readParams(masterFile, plainParamsFile)

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
subField = '';
value = '';


par.inputImageFName = plainParams(executablePath, get_set, masterFile, 'inputImageFName', 	subField, value, '-r');
par.outputImageFName = plainParams(executablePath, get_set, masterFile, 'outputImageFName', 	subField, value, '-r');

par.inputImageListList = plainParams(executablePath, get_set, masterFile, 'inputImageListList', 		subField, value, '-r');
par.outputImageListList = plainParams(executablePath, get_set, masterFile, 'outputImageListList', 	subField, value, '-r');

par.patchPositionList = plainParams(executablePath, get_set, masterFile, 'patchPositionList', 	subField, value, '-r');
par.processIndexList = plainParams(executablePath, get_set, masterFile, 'processIndexList', 	subField, value, '-r');


par.folderSuffix = plainParams(executablePath, get_set, masterFile, 'folderSuffix', 		subField, value, '');

par.maxChunkSize = str2num(	plainParams(executablePath, get_set, masterFile, 'maxChunkSize', 		subField, value, ''));
par.haloRadius = str2num(	plainParams(executablePath, get_set, masterFile, 'haloRadius', 			subField, value, ''));

par.dataType = plainParams(executablePath, get_set, masterFile, 'dataType', 			subField, value, '');

par.maxNumProcesses = str2num(	plainParams(executablePath, get_set, masterFile, 'maxNumProcesses', 		subField, value, ''));

par.isCleanStart = str2num(	plainParams(executablePath, get_set, masterFile, 'isCleanStart', 		subField, value, ''));
par.isCleanEnd = str2num(	plainParams(executablePath, get_set, masterFile, 'isCleanEnd', 		subField, value, ''));


end

