function time = CADMM_denoising_dec( control_4D )


tic

consensus_distribute_dec( control_4D.moduleIOFnames.denoiser_dec_noisy_list, ...
    control_4D.binaryFnames_C.x_cent_prior, control_4D.binaryFnames_C.u_dec_Prior, ...
    1, -1, control_4D.params_consensus.consensus_mode, ...
    control_4D.numVols_denoising, control_4D.num_decentral_denoisers );

for n=1:control_4D.num_decentral_denoisers

    % Denoise
    disp([control_4D.DenoisingConfigList_decentral(n).Script, ' ', control_4D.DenoisingConfigList_decentral(n).ScriptArg]);
    system([control_4D.DenoisingConfigList_decentral(n).Script, ' ', control_4D.DenoisingConfigList_decentral(n).ScriptArg]);

    % Read denoised output
    for t=1:control_4D.numVols_denoising
        vol = read3D( control_4D.moduleIOFnames.denoiser_dec_denoised_list{n,t}, 'float32');
        write3D( control_4D.binaryFnames_C.x_dec_Prior{n,t}, vol, 'float32');
    end

end


time = toc;

return
