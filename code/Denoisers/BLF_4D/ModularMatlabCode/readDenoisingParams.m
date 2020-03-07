function [ par ] = readDenoisingParams( masterFile, plainParamsFile )

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'params';
value = '';
resolveFlag = '';


par.sigmaS = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'sigmaS', value, resolveFlag));
par.sigmaR = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'sigmaR', value, resolveFlag));
par.samS = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'samS', value, resolveFlag));
par.numBins = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'numBins', value, resolveFlag));






end

