function [ newScans, newPar ] = downscaleScansAndParams( scans, par , preprocessingParams)

factor1 = preprocessingParams.downscale_dw;
factor2 = preprocessingParams.downscale_dv;

if(factor1 == 1 && factor2 == 1)
	disp('   (Skip Downscaling)')
	newScans = scans;
	newPar = par;
	return
end

newScans.darkmeanImg = subResolution(scans.darkmeanImg, factor1, factor2, 'sum');
newScans.darkvarImg = subResolution(scans.darkvarImg, factor1, factor2, 'sum');

newScans.blankmeanImg = subResolution(scans.blankmeanImg, factor1, factor2, 'sum');
newScans.blankvarImg = subResolution(scans.blankvarImg, factor1, factor2, 'sum');

newScans.driftReferenceStart_scan = subResolution(scans.driftReferenceStart_scan, factor1, factor2, 'sum');
newScans.driftReferenceEnd_scan = subResolution(scans.driftReferenceEnd_scan, factor1, factor2, 'sum');

newScans.defectivePixelMap = subResolution(scans.defectivePixelMap, factor1, factor2, 'mean');
newScans.defectivePixelMap = double( newScans.defectivePixelMap > 0.5 );

%
newN_dw = size(newScans.blankmeanImg, 1);
newN_dv = size(newScans.blankmeanImg, 2);

newScans.object = zeros(newN_dw, newN_dv, par.N_beta);

for i=1:size(newScans.object,3)
    newScans.object(:,:,i) = subResolution(scans.object(:,:,i), factor1, factor2, 'sum');   
end

for i=1:size(scans.jig_scan,3)
    newScans.jig_scan(:,:,i) = subResolution(scans.jig_scan(:,:,i), factor1, factor2, 'sum');    
end

newPar = par;

newPar.N_dw = newN_dw;
newPar.N_dv = newN_dv;
newPar.Delta_dw = par.Delta_dw * factor1;
newPar.Delta_dv = par.Delta_dv * factor2;

end

