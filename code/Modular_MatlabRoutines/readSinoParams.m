function [ par ] = readSinoParams( masterFile, plainParamsFile );

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'sinoParams';
resolveFlag = '';
value = '';

subField = 'N_dv';
par.N_dv = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'N_dw';
par.N_dw = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'N_beta';
par.N_beta = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'Delta_dv';
par.Delta_dv = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'Delta_dw';
par.Delta_dw = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'u_s';
par.u_s = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'u_r';
par.u_r = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'v_r';
par.v_r = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'u_d0';
par.u_d0 = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'v_d0';
par.v_d0 = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'w_d0';
par.w_d0 = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));



end

