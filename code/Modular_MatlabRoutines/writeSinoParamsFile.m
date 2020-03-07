function [ ] = writeSinoParamsFile( par, masterFile, plainParamsFile );

executablePath = plainParamsFile;
get_set = 'set';
masterField = 'sinoParams';
resolveFlag = '';

value = num2str(par.N_dv);
subField = 'N_dv';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.N_dw);
subField = 'N_dw';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.N_beta);
subField = 'N_beta';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.Delta_dv);
subField = 'Delta_dv';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.Delta_dw);
subField = 'Delta_dw';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.u_s);
subField = 'u_s';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.u_r);
subField = 'u_r';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.v_r);
subField = 'v_r';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.u_d0);
subField = 'u_d0';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.v_d0);
subField = 'v_d0';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);

value = num2str(par.w_d0);
subField = 'w_d0';
plainParams(executablePath, get_set, masterFile, masterField, subField, value, resolveFlag);



end

