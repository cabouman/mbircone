function [ sino, driftReference_sino, jigMeasurementsSino, wght ] = computeSinoAndWeight( scans, par, preprocessingParams)


%%

N_dv = par.N_dv;
N_dw = par.N_dw;
N_beta = par.N_beta;


% preallocate
sino = zeros(N_dw, N_dv, N_beta);
wght = sino;

for i = 1:N_beta
    % v = (l - d) / (b - d)
    % y = - log v

    sino(:,:,i) = photoncount2sino( scans.object(:,:,i), scans.darkmeanImg, scans.blankmeanImg );
end

driftReference_sino = zeros(N_dw, N_dv, 2);
driftReference_sino(:,:,1) = photoncount2sino( scans.driftReferenceStart_scan, scans.darkmeanImg, scans.blankmeanImg );
driftReference_sino(:,:,2) = photoncount2sino( scans.driftReferenceEnd_scan, scans.darkmeanImg, scans.blankmeanImg );

jigMeasurementsSino = zeros(size(scans.jig_scan));
for i = 1:size(scans.jig_scan,3)
    jigMeasurementsSino(:,:,i) = photoncount2sino( scans.jig_scan(:,:,i), scans.darkmeanImg, scans.blankmeanImg );
end


for i = 1:N_beta
    % wght = (l - d).^2 / (l + sigma^2) 

    % wght(:,:,i) = (scans.object(:,:,i) - scans.darkmeanImg).^2 ./ (scans.object(:,:,i) + scans.darkvarImg);
    v = normalize_scan( scans.object(:,:,i), scans.darkmeanImg, scans.blankmeanImg ) ;
    v(v<0) = 0;
    wght(:,:,i) = v;
    
end

wght = normalize_wght(wght, preprocessingParams);


isnan_wt = isnan(wght);
if sum(isnan_wt(:))~=0
    wght(isnan_wt) = 0;
    fprintf('_______________________________________________________________\n');
    fprintf('number of NAN entries in weight set to zero : %d\n',sum(isnan_wt(:)));
    fprintf('_______________________________________________________________\n');
end


end

