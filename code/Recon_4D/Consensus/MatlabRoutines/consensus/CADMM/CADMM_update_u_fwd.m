function time = CADMM_update_u_fwd( control_4D )


tic

for t=1:control_4D.numVols_inversion

	u_fwd = read3D( control_4D.binaryFnames_C.u_fwd{t} , 'float32' );
	x_fwd = read3D( control_4D.binaryFnames_C.x_fwd{t} , 'float32' );
	
	switch control_4D.params_consensus.consensus_mode
	case 'time'
		x_cent_prior = read3D( control_4D.binaryFnames_C.x_cent_prior{t}, 'float32');
	case 'view'
		x_cent_prior = read3D( control_4D.binaryFnames_C.x_cent_prior{1}, 'float32');
	end

	u_fwd = u_fwd + x_fwd - x_cent_prior ;

	write3D( control_4D.binaryFnames_C.u_fwd{t}, u_fwd , 'float32');

end    

time = toc;


