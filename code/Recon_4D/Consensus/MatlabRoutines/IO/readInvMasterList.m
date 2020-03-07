function [ inversionMasterList ] = readInvMasterList( masterFile, plainParamsFile )

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'invMasterList';
value = '';
resolveFlag = '-r';

inversionMasterList_fName = plainParams(executablePath, get_set, masterFile, masterField, '', value, resolveFlag);

inversionMasterList = readFileList(inversionMasterList_fName);

end