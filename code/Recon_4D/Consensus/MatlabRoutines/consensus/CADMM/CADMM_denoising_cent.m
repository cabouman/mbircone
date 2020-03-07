function [time, diffVal] = CADMM_denoising_cent( control_4D )

tic

if ~isempty(control_4D.DenoisingConfig_central.DenoisingConfigFname)

    for t=1:control_4D.numVols_denoising
        x = read3D( control_4D.binaryFnames_C.x_avg{t}, 'float32');
        u = read3D( control_4D.binaryFnames_C.u_avg{t}, 'float32');
        vol = x+u;
        write3D( control_4D.moduleIOFnames.denoiser_cent_noisy_list{t}, vol, 'float32');
    end

    % Denoise
    disp([control_4D.DenoisingConfig_central.Script, ' ', control_4D.DenoisingConfig_central.ScriptArg]);
    system([control_4D.DenoisingConfig_central.Script, ' ', control_4D.DenoisingConfig_central.ScriptArg]);


    % Read denoised output
    for t=1:control_4D.numVols_denoising
        vol = read3D( control_4D.moduleIOFnames.denoiser_cent_denoised_list{t}, 'float32');
        write3D( control_4D.binaryFnames_C.x_cent_prior{t}, vol, 'float32');
    end

    diffVal = stat_RMSE_4D( control_4D.moduleIOFnames.denoiser_cent_denoised_list, control_4D.moduleIOFnames.denoiser_cent_noisy_list );

else
    % copy volumes
    for t=1:control_4D.numVols_denoising
        x = read3D( control_4D.binaryFnames_C.x_avg{t}, 'float32');
        u = read3D( control_4D.binaryFnames_C.u_avg{t}, 'float32');
        vol = x+u;
        write3D( control_4D.binaryFnames_C.x_cent_prior{t}, vol, 'float32');
    end

    diffVal = 0;
    
end


time = toc;

return
