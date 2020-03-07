function [ DenoisingConfigList_decentral ] = readDenoisingConfigList_decentral( masterFile, plainParamsFile )

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
value = '';
resolveFlag = '-r';

DenoisingConfigFnameList_fName = plainParams(executablePath, get_set, masterFile, 'DenoisingConfigList_decentral', '', value, resolveFlag);
DenoisingConfigFnameList = readFileList(DenoisingConfigFnameList_fName);

num_DenoisingConfigs = length(DenoisingConfigFnameList);


if num_DenoisingConfigs==0
	DenoisingConfigList_decentral = [];
else
	for i=1:num_DenoisingConfigs

		DenoisingConfigFname = DenoisingConfigFnameList{i};

		denoisingConfig_temp.DenoisingConfigFname = DenoisingConfigFname;

		denoisingConfig_temp.Script = plainParams(executablePath, get_set, DenoisingConfigFname, 'Script', '', value, resolveFlag);

		denoisingConfig_temp.ScriptArg = plainParams(executablePath, get_set, DenoisingConfigFname, 'ScriptArg', '', value, resolveFlag);

		denoisingConfig_temp.iopath = plainParams(executablePath, get_set, DenoisingConfigFname, 'iopath', '', value, resolveFlag);

		DenoisingConfigList_decentral(i) = denoisingConfig_temp;
	end
end

return