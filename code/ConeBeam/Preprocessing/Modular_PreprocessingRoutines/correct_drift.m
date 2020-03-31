function [ sino, wght, driftSino ] = correct_drift( sino, wght, driftReference_sino, TotalAngle, viewAngleList, preprocessingParams)

N_beta = size(sino,3);
driftSino = zeros(size(sino));

drift_done = 0;

if preprocessingParams.driftCorrection_type == 0
    drift_done = 1;
end


if preprocessingParams.driftCorrection_type == 1
    drift_done = 1;
    % drift correction using background patches
    for i = 1:N_beta
        sino_view = sino(:,:,i);
        sino_view_background = eval([ 'sino_view(' , preprocessingParams.backgroundPatchLimits , ')' ]);
        drift(i) = mean(sino_view_background(:));
    end

    % Subtract drift
    for i = 1:N_beta
        driftSino(:,:,i) = drift(i)*ones(size(sino,1),size(sino,2));
    end
    sino = sino - driftSino;
    wght = wght .* exp(driftSino);

end


if preprocessingParams.driftCorrection_type == 2 || preprocessingParams.driftCorrection_type == 3
    drift_done = 1;
    % drift correction using a reference view (same angle as first view)
    drift_total = driftReference_sino(:,:,2) - driftReference_sino(:,:,1);
    spatial_sigma = preprocessingParams.drift_spatialSigma * [1/preprocessingParams.downscale_dv 1/preprocessingParams.downscale_dw];
    drift_total_lowfreq = imgaussfilt(drift_total, spatial_sigma, 'Padding', 'replicate');

    % Subtract drift
    TotalAngle_rad = (2*pi/360) * TotalAngle ;
    for i = 1:N_beta
        driftSino(:,:,i) = drift_total_lowfreq * viewAngleList(i) / TotalAngle_rad ;
    end
    sino = sino - driftSino;
    wght = wght .* exp(driftSino);

end

if preprocessingParams.driftCorrection_type == 3
    drift_done = 1;
    % Fixing constant offset of sino
    for i = 1:N_beta
        sino_view = sino(:,:,i);
        sino_view_background = eval([ 'sino_view(' , preprocessingParams.backgroundPatchLimits , ')' ]);
        drift(i) = mean(sino_view_background(:));
    end
    mean_offset = mean(drift);
    fprintf('Mean offset in sino: %d \n',mean_offset );
    sino = sino - mean_offset;
    driftSino = driftSino + mean_offset;
    wght = wght .* exp(mean_offset);
    
end

if drift_done == 0
    error('correct_drift: preprocessingParams.driftCorrection_type wrong');
end

return