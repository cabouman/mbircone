function par = convert_sysParams_to_MBIRformat_manual(sysParams, indexList, preprocessingParams, dataSetInfo, scans)

par.u_s = sysParams.u_s ;
par.u_d0 = sysParams.u_d1 ;
par.Delta_dv = sysParams.Delta_dv ;
par.Delta_dw = sysParams.Delta_dw ;
par.TotalAngle = sysParams.TotalAngle ;

par.viewAngleList = computeAngleList( indexList, sysParams.numAcquiredScans, par.TotalAngle, preprocessingParams.rotationDirection );
par.N_beta = length(indexList);



% v & w stuff

par.N_dw = size(scans.object,1);
par.N_dv = size(scans.object,2);


if dataSetInfo.flipParams_v_d1_w_d1==1
	[ sysParams.v_d1 , sysParams.w_d1] = swap(sysParams.v_d1 , sysParams.w_d1) ;
end

switch dataSetInfo.flip_v_d0
case 0
	par.v_d0 = -sysParams.v_d1 ;
case 1
	par.v_d0 = sysParams.v_d1 - par.N_dv * par.Delta_dv ;
case 0.5
	par.v_d0 =  - par.N_dv * par.Delta_dv / 2;
end

switch dataSetInfo.flip_w_d0
case 0
	par.w_d0 = -sysParams.w_d1 ;
case 1
	par.w_d0 = sysParams.w_d1 - par.N_dw * par.Delta_dw ;
case 0.5
	par.w_d0 =  - par.N_dw * par.Delta_dw / 2;
end


disp('sys params in convert_sysParams_to_MBIRformat_manual');
sysParams
disp('par in convert_sysParams_to_MBIRformat_manual');
par

par.u_r = 0 ;
par.v_r = 0 ;

return



function [A, B] = swap(a, b)

B = a;
A = b;

return