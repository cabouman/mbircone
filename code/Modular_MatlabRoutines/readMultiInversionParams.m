function [ par ] = readMultiInversionParams( masterFile, plainParamsFile )

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
masterField = '';
value = '';
resolveFlag = '';


par.invMasterList =                 plainParams(executablePath, get_set, masterFile, 'invMasterList',       masterField, value, '-r');
par.invMaster =                     plainParams(executablePath, get_set, masterFile, 'invMaster',           masterField, value, '-r');
par.num_timePoints_all =    str2num(plainParams(executablePath, get_set, masterFile, 'num_timePoints_all',  masterField, value, ''));
par.time_start =            str2num(plainParams(executablePath, get_set, masterFile, 'time_start',          masterField, value, ''));
par.time_end =              str2num(plainParams(executablePath, get_set, masterFile, 'time_end',            masterField, value, ''));
par.time_step =             str2num(plainParams(executablePath, get_set, masterFile, 'time_step',           masterField, value, ''));
par.num_viewSubsets =       str2num(plainParams(executablePath, get_set, masterFile, 'num_viewSubsets',     masterField, value, ''));
par.prefixRoot =                    plainParams(executablePath, get_set, masterFile, 'prefixRoot',          masterField, value, '');
par.suffixRoot =                    plainParams(executablePath, get_set, masterFile, 'suffixRoot',          masterField, value, '');
par.multiNodeMaster =               plainParams(executablePath, get_set, masterFile, 'multiNodeMaster',     masterField, value, '-r');













return
