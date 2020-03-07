function [ inversionConfig ] = readInversionConfig( masterFile, plainParamsFile )

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'inversionConfig';
value = '';
resolveFlag = '-r';


inversionConfig.recon_Script_4D = plainParams(executablePath, get_set, masterFile, masterField, 'recon_Script_4D', value, resolveFlag);

resolveFlag = '';

inversionConfig.recon_4D_mode_init = plainParams(executablePath, get_set, masterFile, masterField, 'recon_4D_mode_init', value, resolveFlag);

inversionConfig.recon_4D_mode_proxmap = plainParams(executablePath, get_set, masterFile, masterField, 'recon_4D_mode_proxmap', value, resolveFlag);


end