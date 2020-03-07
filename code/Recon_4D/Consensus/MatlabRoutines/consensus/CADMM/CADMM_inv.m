function time = CADMM_inv( control_4D )


tic

% Write proxmap input 
consensus_distribute_fwd( control_4D.moduleIOFnames.inv_input_list, ...
    control_4D.binaryFnames_C.x_cent_prior, control_4D.binaryFnames_C.u_fwd, ...
    1, -1, control_4D.params_consensus.consensus_mode, ...
    control_4D.numVols_inversion );

% Do inversion
system([control_4D.inversionConfig.recon_Script_4D, ' ', control_4D.master_4D, ' ', control_4D.inversionConfig.recon_4D_mode_proxmap]);

% Read from inversion
for t=1:control_4D.numVols_inversion
    vol = read3D( control_4D.moduleIOFnames.inv_output_list{t}, 'float32');
    write3D( control_4D.binaryFnames_C.x_fwd{t}, vol , 'float32');
end

time = toc;

return 

