function [control_4D] = CADMM_step(control_4D, itnumber)

disp(['================','ADMM Iteration: ', num2str(itnumber),  '====================================================================']);
%% x_fwd_before_t = x_fwd_t     t=1:T
for t=1:control_4D.numVols_inversion
    x = read3D( control_4D.binaryFnames_C.x_fwd{t}, 'float32' );
    write3D( control_4D.binaryFnames_C.x_fwd_before{t}, x , 'float32');
end

if(itnumber~=0)

    control_4D.stats.iters_done(itnumber) = itnumber;

    if control_4D.params_consensus.testDenoiser ~= 1
        disp(' === CADMM_inv ===');
        % x_fwd = F(x_cent_prior-u_fwd)
        control_4D.stats.time.inv(itnumber) = CADMM_inv( control_4D );
    end

    % x_dec_Prior = H1(x_cent_prior-u_dec_Prior)
    disp(' === CADMM_denoising_dec ===')
    control_4D.stats.time.denoising_dec(itnumber) = CADMM_denoising_dec( control_4D );

    % time mode:
    %   x_avg_t = avg(x_fwd_t,x_dec_Prior_t) 
    % view mode:
    %   x_avg = avg(x_fwd_t,x_dec_Prior_n)  over volumes index
    disp(' === CADMM: avg of x ===')
    control_4D.stats.time.avg_x(itnumber) = consensus_compute_avg( control_4D.binaryFnames_C.x_avg, ...
        control_4D.binaryFnames_C.x_fwd, ...
        control_4D.binaryFnames_C.x_dec_Prior, ...
        control_4D.params_consensus.consensus_mode, control_4D.numVols_inversion, control_4D.num_decentral_denoisers, ...
        control_4D.params_consensus.averaging_wtList );


    % time mode:
    %   u_avg_t = avg(u_fwd_t,u_dec_Prior_t) 
    % view mode:
    %   u_avg = avg(u_fwd_t,u_dec_Prior_n)  over volumes index
    disp(' === CADMM: avg of x ===')
    control_4D.stats.time.avg_u(itnumber) = consensus_compute_avg( control_4D.binaryFnames_C.u_avg, ...
        control_4D.binaryFnames_C.u_fwd, ...
        control_4D.binaryFnames_C.u_dec_Prior, ...
        control_4D.params_consensus.consensus_mode, control_4D.numVols_inversion, control_4D.num_decentral_denoisers, ...
        control_4D.params_consensus.averaging_wtList );


    % x_cent_prior = denoise(x_avg+u_avg)
    disp(' === CADMM_denoising_cent ===')
    [control_4D.stats.time.denoising_cent(itnumber), control_4D.stats.diff_io_denoising_cent(itnumber)] = CADMM_denoising_cent( control_4D );

    % u_fwd = u_fwd + (x_fwd-x_cent_prior)
    disp(' === CADMM_update_u_fwd ===')
    control_4D.stats.time.update_u_fwd(itnumber) = CADMM_update_u_fwd( control_4D );

    % u_dec_Prior = u_dec_Prior + (x_dec_Prior-x_cent_prior)
    disp(' === CADMM_update_u_dec_Prior ===')
    control_4D.stats.time.update_u_dec_Prior(itnumber) = CADMM_update_u_dec_Prior( control_4D );

end

return
