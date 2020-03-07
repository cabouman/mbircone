function [ binaryFNames ] = readBinaryFNames( masterFile, plainParamsFile )


% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);



executablePath = plainParamsFile;
get_set = 'get';
masterField = 'binaryFNames';
value = '';
resolveFlag = '-r';

binaryFNames.sino = plainParams(executablePath, get_set, masterFile, masterField, 'sino', value, resolveFlag);

binaryFNames.driftSino = plainParams(executablePath, get_set, masterFile, masterField, 'driftSino', value, resolveFlag);

binaryFNames.origSino = plainParams(executablePath, get_set, masterFile, masterField, 'origSino', value, resolveFlag);

binaryFNames.wght = plainParams(executablePath, get_set, masterFile, masterField, 'wght', value, resolveFlag);

binaryFNames.errsino = plainParams(executablePath, get_set, masterFile, masterField, 'errsino', value, resolveFlag);

binaryFNames.sinoMask = plainParams(executablePath, get_set, masterFile, masterField, 'sinoMask', value, resolveFlag);

binaryFNames.recon = plainParams(executablePath, get_set, masterFile, masterField, 'recon', value, resolveFlag);

binaryFNames.reconROI = plainParams(executablePath, get_set, masterFile, masterField, 'reconROI', value, resolveFlag);

binaryFNames.proxMapInput = plainParams(executablePath, get_set, masterFile, masterField, 'proxMapInput', value, resolveFlag);

binaryFNames.lastChange = plainParams(executablePath, get_set, masterFile, masterField, 'lastChange', value, resolveFlag);

binaryFNames.timeToChange = plainParams(executablePath, get_set, masterFile, masterField, 'timeToChange', value, resolveFlag);

binaryFNames.phantom = plainParams(executablePath, get_set, masterFile, masterField, 'phantom', value, resolveFlag);

binaryFNames.sysMatrix = plainParams(executablePath, get_set, masterFile, masterField, 'sysMatrix', value, resolveFlag);

binaryFNames.wghtRecon = plainParams(executablePath, get_set, masterFile, masterField, 'wghtRecon', value, resolveFlag);




end







