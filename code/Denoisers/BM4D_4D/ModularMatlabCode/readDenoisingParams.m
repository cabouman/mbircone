function [ par ] = readDenoisingParams( masterFile, plainParamsFile )

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'params';
value = '';
resolveFlag = '';


par.estimate_sigma =	str2num(plainParams(executablePath, get_set, masterFile, masterField, 'estimate_sigma',	value, resolveFlag));
par.sigma = 			str2num(plainParams(executablePath, get_set, masterFile, masterField, 'sigma', 			value, resolveFlag));
par.distribution = 			   (plainParams(executablePath, get_set, masterFile, masterField, 'distribution', 	value, resolveFlag));
par.profile = 				   (plainParams(executablePath, get_set, masterFile, masterField, 'profile', 		value, resolveFlag));
par.do_wiener = 		str2num(plainParams(executablePath, get_set, masterFile, masterField, 'do_wiener', 		value, resolveFlag));
par.verbose = 			str2num(plainParams(executablePath, get_set, masterFile, masterField, 'verbose', 		value, resolveFlag));

par.searchWindowSize = 	str2num(plainParams(executablePath, get_set, masterFile, masterField, 'searchWindowSize', 		value, resolveFlag));
if(isempty(par.searchWindowSize))
	par.searchWindowSize = 11;
end

par.const_dims_4D = 		str2num(plainParams(executablePath, get_set, masterFile, masterField, 'const_dims_4D', 	value, resolveFlag));







end

