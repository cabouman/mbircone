function setReconParamsField(fieldString, valueString, masterFile, plainParamsFile)

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'set';
masterField = 'reconParams';
subField = fieldString;
value = valueString;
resolveFlag = '';

plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);






end