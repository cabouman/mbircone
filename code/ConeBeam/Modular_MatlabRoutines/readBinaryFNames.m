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

binaryFNames.estimateSino = plainParams(executablePath, get_set, masterFile, masterField, 'estimateSino', value, resolveFlag);

binaryFNames.errSino = plainParams(executablePath, get_set, masterFile, masterField, 'errSino', value, resolveFlag);

binaryFNames.wght = plainParams(executablePath, get_set, masterFile, masterField, 'wght', value, resolveFlag);

binaryFNames.origSino = plainParams(executablePath, get_set, masterFile, masterField, 'origSino', value, resolveFlag);

binaryFNames.jigMeasurementsSino = plainParams(executablePath, get_set, masterFile, masterField, 'jigMeasurementsSino', value, resolveFlag);

binaryFNames.driftSino = plainParams(executablePath, get_set, masterFile, masterField, 'driftSino', value, resolveFlag);

binaryFNames.projOutput = plainParams(executablePath, get_set, masterFile, masterField, 'projOutput', value, resolveFlag);

binaryFNames.recon = plainParams(executablePath, get_set, masterFile, masterField, 'recon', value, resolveFlag);

binaryFNames.reconROI = plainParams(executablePath, get_set, masterFile, masterField, 'reconROI', value, resolveFlag);

binaryFNames.proxMapInput = plainParams(executablePath, get_set, masterFile, masterField, 'proxMapInput', value, resolveFlag);

binaryFNames.lastChange = plainParams(executablePath, get_set, masterFile, masterField, 'lastChange', value, resolveFlag);

binaryFNames.phantom = plainParams(executablePath, get_set, masterFile, masterField, 'phantom', value, resolveFlag);

binaryFNames.wghtRecon = plainParams(executablePath, get_set, masterFile, masterField, 'wghtRecon', value, resolveFlag);

binaryFNames.projInput = plainParams(executablePath, get_set, masterFile, masterField, 'projInput', value, resolveFlag);

binaryFNames.consensusRecon = plainParams(executablePath, get_set, masterFile, masterField, 'consensusRecon', value, resolveFlag);

binaryFNames.sysMatrix = plainParams(executablePath, get_set, masterFile, masterField, 'sysMatrix', value, resolveFlag);

binaryFNames.timeToChange = plainParams(executablePath, get_set, masterFile, masterField, 'timeToChange', value, resolveFlag);

binaryFNames.backprojlikeInput = plainParams(executablePath, get_set, masterFile, masterField, 'backprojlikeInput', value, resolveFlag);

binaryFNames.backprojlikeOutput = plainParams(executablePath, get_set, masterFile, masterField, 'backprojlikeOutput', value, resolveFlag);


end