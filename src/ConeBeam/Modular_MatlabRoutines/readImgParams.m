function [ par ] = readImgParams( masterFile, plainParamsFile );

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'imgParams';
resolveFlag = '';
value = '';

subField = 'x_0';
par.x_0 = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'y_0';
par.y_0 = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'z_0';
par.z_0 = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'N_x';
par.N_x = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'N_y';
par.N_y = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'N_z';
par.N_z = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'Delta_xy';
par.Delta_xy = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'Delta_z';
par.Delta_z = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'j_xstart_roi';
par.j_xstart_roi = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'j_ystart_roi';
par.j_ystart_roi = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'j_zstart_roi';
par.j_zstart_roi = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'j_xstop_roi';
par.j_xstop_roi = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'j_ystop_roi';
par.j_ystop_roi = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));

subField = 'j_zstop_roi';
par.j_zstop_roi = str2num(plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag));


end

