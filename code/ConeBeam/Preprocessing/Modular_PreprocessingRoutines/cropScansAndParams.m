function [ newScans, newPar ] = cropScansAndParams( scans, par, preprocessingParams )

crop_dv0 = preprocessingParams.crop_dv0;
crop_dv1 = preprocessingParams.crop_dv1;
crop_dw0 = preprocessingParams.crop_dw0;
crop_dw1 = preprocessingParams.crop_dw1;

if(		crop_dv0 < 0 ...
	|| 	crop_dv0 >= crop_dv1 ...
	||	crop_dv1 > 1 ...
	||	crop_dw0 < 0 ...
	|| 	crop_dw0 >= crop_dw1 ...
	||	crop_dw1 > 1)
	error('Preprocessing params invalid. Need (0 <= crop_dv0 < crop_dv1 <= 1) and (0 <= crop_dw0 < crop_dw1 <= 1)');
end

if(		crop_dv0 == 0 ...
	&&	crop_dv1 == 1 ...
	&&	crop_dw0 == 0 ...
	&&	crop_dw1 == 1)
	disp('   (Skip Cropping)')
	newScans = scans;
	newPar = par;
	return
end

N_dvshift0 = round(par.N_dv * crop_dv0);
N_dvshift1 = round(par.N_dv * (1-crop_dv1));
v_d0_new = par.v_d0 + N_dvshift0 * par.Delta_dv;
N_dv_new = par.N_dv - (N_dvshift0 + N_dvshift1);

N_dwshift0 = round(par.N_dw * crop_dw0);
N_dwshift1 = round(par.N_dw * (1-crop_dw1));
w_d0_new = par.w_d0 + N_dwshift0 * par.Delta_dw;
N_dw_new = par.N_dw - (N_dwshift0 + N_dwshift1);


%% Crop photon count data

newScans.darkmeanImg = cropScan( scans.darkmeanImg, N_dwshift0, N_dwshift1, N_dvshift0, N_dvshift1 );
newScans.darkvarImg = cropScan( scans.darkvarImg, N_dwshift0, N_dwshift1, N_dvshift0, N_dvshift1 );

newScans.blankmeanImg = cropScan( scans.blankmeanImg, N_dwshift0, N_dwshift1, N_dvshift0, N_dvshift1 );
newScans.blankvarImg = cropScan( scans.blankvarImg, N_dwshift0, N_dwshift1, N_dvshift0, N_dvshift1 );

newScans.driftReferenceStart_scan = cropScan( scans.driftReferenceStart_scan, N_dwshift0, N_dwshift1, N_dvshift0, N_dvshift1 );
newScans.driftReferenceEnd_scan = cropScan( scans.driftReferenceEnd_scan, N_dwshift0, N_dwshift1, N_dvshift0, N_dvshift1 );

newScans.defectivePixelMap = cropScan( scans.defectivePixelMap, N_dwshift0, N_dwshift1, N_dvshift0, N_dvshift1 );

newScans.object = zeros( N_dw_new, N_dv_new, par.N_beta );
for i_beta=1:size(newScans.object,3)
    newScans.object(:,:,i_beta) = cropScan( scans.object(:,:,i_beta), N_dwshift0, N_dwshift1, N_dvshift0, N_dvshift1 );
end

newScans.occlusion_scan = zeros( N_dw_new, N_dv_new, size(scans.occlusion_scan,3) );
for i=1:size(newScans.occlusion_scan,3)
    newScans.occlusion_scan(:,:,i) = cropScan( scans.occlusion_scan(:,:,i), N_dwshift0, N_dwshift1, N_dvshift0, N_dvshift1 );
end


%%

%% updating parameter structure
newPar = par;

newPar.w_d0 = w_d0_new;
newPar.N_dw = N_dw_new;

newPar.v_d0 = v_d0_new;
newPar.N_dv = N_dv_new;

end

