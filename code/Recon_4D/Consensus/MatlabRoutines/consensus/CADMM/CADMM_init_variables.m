function time = CADMM_init_variables( control_4D )


%       Time:
%
%       x_fwd_t = inversion_qggmrf(y_t)     t=1:T
%
%       x_dec_Prior_t_n = x_fwd_t           t=1:T n=1:N
%       x_avg_t = x_fwd_t                   t=1:T
%       x_cent_prior_t = x_fwd_t            t=1:T
%
%       u_avg_t = 0                         t=1:T
%       u_fwd_t = 0                         t=1:T
%       u_dec_Prior_t = 0                   t=1:T n=1:N
% 

%       View subsets:
%
%       x_fwd_t = inversion_qggmrf(y_t)     t=1:T
%
%       x_dec_Prior_n = mean(  x_fwd_t )    n=1:N
%       x_avg = mean(  x_fwd_t )      
%       x_cent_prior = x_avg 
%
%       u_avg = 0                         
%       u_fwd_t = 0                         t=1:T
%       u_dec_Prior = 0                     n=1:N
%

if control_4D.numVols_inversion < control_4D.numVols_denoising
    error('CADMM_init_variables: error numVols_inversion < numVols_denoising');
end

tic

control_4D.std_inv_list = zeros(1,control_4D.numVols_inversion);

% x_fwd
for t=1:control_4D.numVols_inversion
    x = read3D( control_4D.moduleIOFnames.inv_output_list{t}, 'float32');
    std_inv_list(t) = std(x(:));
    write3D( control_4D.binaryFnames_C.x_fwd{t}, x , 'float32');
end

% u's
zero_vol = 0*read3D( control_4D.binaryFnames_C.x_fwd{1}, 'float32');
for t=1:control_4D.numVols_inversion
    write3D( control_4D.binaryFnames_C.u_fwd{t}, zero_vol , 'float32');
end
for t=1:control_4D.numVols_denoising
    write3D( control_4D.binaryFnames_C.u_avg{t}, zero_vol , 'float32');
    for n=1:control_4D.num_decentral_denoisers
        write3D( control_4D.binaryFnames_C.u_dec_Prior{n,t}, zero_vol , 'float32');
    end
end


% x_avg
consensus_compute_avg( control_4D.binaryFnames_C.x_avg, ...
                    control_4D.binaryFnames_C.x_fwd, ...
                    [], ...
                    control_4D.params_consensus.consensus_mode, control_4D.numVols_inversion, 0, [1] );



for t=1:control_4D.numVols_denoising
    x = read3D( control_4D.binaryFnames_C.x_avg{t}, 'float32');
    write3D( control_4D.binaryFnames_C.x_cent_prior{t}, x , 'float32');
    for n=1:control_4D.num_decentral_denoisers
        write3D( control_4D.binaryFnames_C.x_dec_Prior{n,t}, x , 'float32');
    end
end


time = toc;