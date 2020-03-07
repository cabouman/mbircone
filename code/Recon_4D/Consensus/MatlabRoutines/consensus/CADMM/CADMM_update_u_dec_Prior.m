function time = CADMM_update_u_dec_Prior( control_4D )


tic
for n=1:control_4D.num_decentral_denoisers
	for t=1:control_4D.numVols_denoising

		u_dec_Prior = read3D( control_4D.binaryFnames_C.u_dec_Prior{n,t} , 'float32' );
		x_dec_Prior = read3D( control_4D.binaryFnames_C.x_dec_Prior{n,t} , 'float32' );


		switch control_4D.params_consensus.consensus_mode
		case 'time'
			x_cent_prior = read3D( control_4D.binaryFnames_C.x_cent_prior{t}, 'float32');
		case 'view'
			x_cent_prior = read3D( control_4D.binaryFnames_C.x_cent_prior{1}, 'float32');
		end

		u_dec_Prior = u_dec_Prior + x_dec_Prior - x_cent_prior ;

		write3D( control_4D.binaryFnames_C.u_dec_Prior{n,t}, u_dec_Prior , 'float32');

	end  
end  

time = toc;

