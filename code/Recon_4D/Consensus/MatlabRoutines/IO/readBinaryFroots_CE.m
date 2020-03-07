function [ binaryFroots_C ] = readBinaryFroots_CE( masterFile, plainParamsFile )


% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'binaryFroots_CE';
value = '';
resolveFlag = '-r';


binaryFroots_C.x_fwd = plainParams(executablePath, get_set, masterFile, masterField, 'x_fwd', value, resolveFlag);
binaryFroots_C.x_fwd_before = plainParams(executablePath, get_set, masterFile, masterField, 'x_fwd_before', value, resolveFlag);
binaryFroots_C.x_dec_Prior = plainParams(executablePath, get_set, masterFile, masterField, 'x_dec_Prior', value, resolveFlag);

binaryFroots_C.v_fwd = plainParams(executablePath, get_set, masterFile, masterField, 'v_fwd', value, resolveFlag);
binaryFroots_C.v_dec_Prior = plainParams(executablePath, get_set, masterFile, masterField, 'v_dec_Prior', value, resolveFlag);
binaryFroots_C.v_avg = plainParams(executablePath, get_set, masterFile, masterField, 'v_avg', value, resolveFlag);

binaryFroots_C.w_fwd = plainParams(executablePath, get_set, masterFile, masterField, 'w_fwd', value, resolveFlag);
binaryFroots_C.w_dec_Prior = plainParams(executablePath, get_set, masterFile, masterField, 'w_dec_Prior', value, resolveFlag);

binaryFroots_C.w_step_fwd = plainParams(executablePath, get_set, masterFile, masterField, 'w_step_fwd', value, resolveFlag);
binaryFroots_C.w_step_dec_Prior = plainParams(executablePath, get_set, masterFile, masterField, 'w_step_dec_Prior', value, resolveFlag);

binaryFroots_C.z_cent_prior = plainParams(executablePath, get_set, masterFile, masterField, 'z_cent_prior', value, resolveFlag);


resolveFlag = '';
binaryFroots_C.fwd_suffix = plainParams(executablePath, get_set, masterFile, masterField, 'fwd_suffix', value, resolveFlag);
binaryFroots_C.dec_Prior_suffix = plainParams(executablePath, get_set, masterFile, masterField, 'dec_Prior_suffix', value, resolveFlag);

return
