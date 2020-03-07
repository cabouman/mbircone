function [] = writeDenoiserSigma(sigma, masterFile, plainParamsFile )

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'set';
masterField = 'params';
subField = 'sigma';
value = num2str(sigma);
resolveFlag = '';

plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

end