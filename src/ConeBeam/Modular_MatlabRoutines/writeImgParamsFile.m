function [ ] = writeImgParamsFile( par, masterFile, plainParamsFile );





executablePath = plainParamsFile;
get_set = 'set';
masterField = 'imgParams';
resolveFlag = '';


value = num2str(par.x_0);
subField = 'x_0';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.y_0);
subField = 'y_0';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.z_0);
subField = 'z_0';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.N_x);
subField = 'N_x';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.N_y);
subField = 'N_y';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.N_z);
subField = 'N_z';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.Delta_xy);
subField = 'Delta_xy';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.Delta_z);
subField = 'Delta_z';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.j_xstart_roi);
subField = 'j_xstart_roi';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.j_ystart_roi);
subField = 'j_ystart_roi';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.j_zstart_roi);
subField = 'j_zstart_roi';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.j_xstop_roi);
subField = 'j_xstop_roi';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.j_ystop_roi);
subField = 'j_ystop_roi';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.j_zstop_roi);
subField = 'j_zstop_roi';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);








end

