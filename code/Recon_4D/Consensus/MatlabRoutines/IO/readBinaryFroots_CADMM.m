function [ binaryFroots_C ] = readBinaryFroots_CADMM( masterFile, plainParamsFile )


% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'binaryFroots_CADMM';
value = '';
resolveFlag = '-r';


binaryFroots_C.x_fwd = plainParams(executablePath, get_set, masterFile, masterField, 'x_fwd', value, resolveFlag);
binaryFroots_C.u_fwd = plainParams(executablePath, get_set, masterFile, masterField, 'u_fwd', value, resolveFlag);
binaryFroots_C.x_dec_Prior = plainParams(executablePath, get_set, masterFile, masterField, 'x_dec_Prior', value, resolveFlag);
binaryFroots_C.u_dec_Prior = plainParams(executablePath, get_set, masterFile, masterField, 'u_dec_Prior', value, resolveFlag);

binaryFroots_C.x_avg = plainParams(executablePath, get_set, masterFile, masterField, 'x_avg', value, resolveFlag);
binaryFroots_C.u_avg = plainParams(executablePath, get_set, masterFile, masterField, 'u_avg', value, resolveFlag);

binaryFroots_C.x_cent_prior = plainParams(executablePath, get_set, masterFile, masterField, 'x_cent_prior', value, resolveFlag);

binaryFroots_C.x_fwd_before = plainParams(executablePath, get_set, masterFile, masterField, 'x_fwd_before', value, resolveFlag);

resolveFlag = '';
binaryFroots_C.fwd_suffix = plainParams(executablePath, get_set, masterFile, masterField, 'fwd_suffix', value, resolveFlag);
binaryFroots_C.dec_Prior_suffix = plainParams(executablePath, get_set, masterFile, masterField, 'dec_Prior_suffix', value, resolveFlag);

return
