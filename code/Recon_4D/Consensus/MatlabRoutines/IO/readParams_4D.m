function [ params_4D ] = readParams_4D( masterFile, plainParamsFile )

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'params_4D';
value = '';
resolveFlag = '';

params_4D.mode_4D  = plainParams(executablePath, get_set, masterFile, masterField, 'mode_4D', value, resolveFlag);

params_4D.num_timePoints  = str2num( plainParams(executablePath, get_set, masterFile, masterField, 'num_timePoints', value, resolveFlag) );
params_4D.time_start  = str2num( plainParams(executablePath, get_set, masterFile, masterField, 'time_start', value, resolveFlag) );
params_4D.time_end  = str2num( plainParams(executablePath, get_set, masterFile, masterField, 'time_end', value, resolveFlag) );
params_4D.time_step  = str2num( plainParams(executablePath, get_set, masterFile, masterField, 'time_step', value, resolveFlag) );

params_4D.num_viewSubsets  = str2num( plainParams(executablePath, get_set, masterFile, masterField, 'num_viewSubsets', value, resolveFlag) );

params_4D.prefixRoot  = plainParams(executablePath, get_set, masterFile, masterField, 'prefixRoot', value, resolveFlag) ;
params_4D.suffixRoot  = plainParams(executablePath, get_set, masterFile, masterField, 'suffixRoot', value, resolveFlag) ;


return
