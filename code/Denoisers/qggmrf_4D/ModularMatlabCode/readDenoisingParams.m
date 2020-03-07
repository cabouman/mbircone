function [ par ] = readDenoisingParams( masterFile, plainParamsFile )

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'params';
value = '';
resolveFlag = '';


par.is_positivity_constraint = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'is_positivity_constraint', value, resolveFlag)) ;

par.q = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'q', value, resolveFlag)) ;
par.p = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'p', value, resolveFlag)) ;

par.T_s = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'T_s', value, resolveFlag)) ;
par.T_t = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'T_t', value, resolveFlag)) ;

par.sigma_s = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'sigma_s', value, resolveFlag)) ;
par.sigma_t = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'sigma_t', value, resolveFlag)) ;
par.sigma = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'sigma', value, resolveFlag)) ;

par.isTimePrior = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'isTimePrior', value, resolveFlag)) ;

par.stopThreshold = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'stopThreshold', value, resolveFlag)) ;
par.MaxIterations = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'MaxIterations', value, resolveFlag)) ;

par.isSaveImage = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'isSaveImage', value, resolveFlag)) ;




end

