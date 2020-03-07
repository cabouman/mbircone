function [ DenoisingConfig_central ] = readDenoisingConfig_central( masterFile, plainParamsFile )

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'DenoisingConfig_central';
value = '';
resolveFlag = '-r';

DenoisingConfigFname = plainParams(executablePath, get_set, masterFile, masterField, '', value, resolveFlag);

DenoisingConfig_central.DenoisingConfigFname = DenoisingConfigFname;

if ~isempty(DenoisingConfig_central.DenoisingConfigFname)

	DenoisingConfig_central.Script = plainParams(executablePath, get_set, masterFile, masterField, 'Script', value, resolveFlag);

	DenoisingConfig_central.ScriptArg = plainParams(executablePath, get_set, masterFile, masterField, 'ScriptArg', value, resolveFlag);

	DenoisingConfig_central.iopath = plainParams(executablePath, get_set, masterFile, masterField, 'iopath', value, resolveFlag);
end

end