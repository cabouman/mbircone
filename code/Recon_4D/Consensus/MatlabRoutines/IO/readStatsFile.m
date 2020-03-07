function [ statsFile ] = readStatsFile( masterFile, plainParamsFile )

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'statsFile';
value = '';
resolveFlag = '-r';


statsFile = plainParams(executablePath, get_set, masterFile, masterField, '', value, resolveFlag);

end