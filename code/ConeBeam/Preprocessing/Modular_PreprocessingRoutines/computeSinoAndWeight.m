function [ sino, driftReference_sino, occlusion_sino, wght ] = computeSinoAndWeight( scans, par, preprocessingParams)


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

driftReferenceStart_sino = photoncount2sino( scans.driftReferenceStart_scan, scans.darkmeanImg, scans.blankmeanImg );
driftReferenceEnd_sino = photoncount2sino( scans.driftReferenceEnd_scan, scans.darkmeanImg, scans.blankmeanImg );
driftReference_sino = zeros(N_dw, N_dv, 2);
driftReference_sino(:,:,1) = driftReferenceStart_sino;
driftReference_sino(:,:,2) = driftReferenceEnd_sino;

occlusion_sino = zeros(size(scans.occlusion_scan));
for i = 1:size(scans.occlusion_scan,3)
    occlusion_sino(:,:,i) = photoncount2sino( scans.occlusion_scan(:,:,i), scans.darkmeanImg, scans.blankmeanImg );
end


for i = 1:N_beta
    % wght = (l - d).^2 / (l + sigma^2) 

    % wght(:,:,i) = (scans.object(:,:,i) - scans.darkmeanImg).^2 ./ (scans.object(:,:,i) + scans.darkvarImg);

    v = normalize_scan( scans.object(:,:,i), scans.darkmeanImg, scans.blankmeanImg ) * max(scans.blankmeanImg(:));
    v(v<0) = 0;
    wght(:,:,i) = v;
    
end

% normalize by number of views so regularization is invariant with respect to num. views
wght = wght / N_beta;

% normalize using number of view subsets. This makes regularization invariant to num. view subsets
% This is because [N_beta_all = N_beta * num_viewSubsets = const.] when varying num_viewSubsets.
wght = wght / preprocessingParams.num_viewSubsets;

% this makes weight scaler approximately invariant to downcaling
wght = wght / sqrt(preprocessingParams.downscale_dv*preprocessingParams.downscale_dw);

isnan_wt = isnan(wght);

if sum(isnan_wt(:))~=0
    wght(isnan_wt) = 0;
    fprintf('_______________________________________________________________\n');
    fprintf('number of NAN entries in weight set to zero : %d\n',sum(isnan_wt(:)));
    fprintf('_______________________________________________________________\n');
end


end

