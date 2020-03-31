function [ par ] = readDataSetInfo_genScan( masterFile, plainParamsFile );

executablePath = plainParamsFile;
get_set = 'get';

masterField = 'dataSetInfo';
resolveFlag = '-r';
value = '';



subField = 'blankScan';
par.blankScan = plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

subField = 'scan';
par.scan = plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

subField = 'params';
par.params = plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);



subMasterFile = par.params;
subField = '';
resolveFlag = '';

par.par.N_dv = str2num(plainParams(executablePath, get_set, subMasterFile, 'N_dv', subField, value, resolveFlag));

par.par.N_dw = str2num(plainParams(executablePath, get_set, subMasterFile, 'N_dw', subField, value, resolveFlag));

par.par.N_beta = str2num(plainParams(executablePath, get_set, subMasterFile, 'N_beta', subField, value, resolveFlag));

par.par.Delta_dv = str2num(plainParams(executablePath, get_set, subMasterFile, 'Delta_dv', subField, value, resolveFlag));

par.par.Delta_dw = str2num(plainParams(executablePath, get_set, subMasterFile, 'Delta_dw', subField, value, resolveFlag));

par.par.u_s = str2num(plainParams(executablePath, get_set, subMasterFile, 'u_s', subField, value, resolveFlag));

par.par.u_r = str2num(plainParams(executablePath, get_set, subMasterFile, 'u_r', subField, value, resolveFlag));

par.par.v_r = str2num(plainParams(executablePath, get_set, subMasterFile, 'v_r', subField, value, resolveFlag));

par.par.u_d0 = str2num(plainParams(executablePath, get_set, subMasterFile, 'u_d0', subField, value, resolveFlag));

par.par.v_d0 = str2num(plainParams(executablePath, get_set, subMasterFile, 'v_d0', subField, value, resolveFlag));

par.par.w_d0 = str2num(plainParams(executablePath, get_set, subMasterFile, 'w_d0', subField, value, resolveFlag));

par.par.TotalAngle = str2num(plainParams(executablePath, get_set, subMasterFile, 'TotalAngle', subField, value, resolveFlag));



end

























