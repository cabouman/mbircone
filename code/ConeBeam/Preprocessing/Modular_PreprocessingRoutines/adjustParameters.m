function [par] = adjustParameters(par, preprocessingParams)

par.u_s  = par.u_s  + preprocessingParams.Delta_u_s ;
par.u_d0 = par.u_d0 + preprocessingParams.Delta_u_d0;
par.v_d0 = par.v_d0 + preprocessingParams.Delta_v_d0;
par.w_d0 = par.w_d0 + preprocessingParams.Delta_w_d0;

% adjusting v_r requires readjusting v_d0
par.v_r  = par.v_r  + preprocessingParams.Delta_v_r;
M = (par.u_d0 - par.u_s)/ (-par.u_s);
par.v_d0 = par.v_d0 + M*preprocessingParams.Delta_v_r;




end