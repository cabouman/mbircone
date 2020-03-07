function time = consensus_distribute_dec( dest_decVol_fNameList, src_centVol_fNameList, src_decVol_fNameList, coeff_cent, coeff_dec, consensus_mode, numVols_denoising, num_decentral_denoisers )

tic

for n=1:num_decentral_denoisers
	for t=1:numVols_denoising

	    switch consensus_mode
	    case 'time'
	        vol_src_cent = read3D( src_centVol_fNameList{t}, 'float32');
	    case 'view'
	        vol_src_cent = read3D( src_centVol_fNameList{1}, 'float32');
	    end

	    vol_src_dec = read3D( src_decVol_fNameList{n,t}, 'float32');

	    vol_dest = coeff_cent*vol_src_cent + coeff_dec*vol_src_dec ;

	    write3D( dest_decVol_fNameList{n,t}, vol_dest , 'float32');
	end
end

time = toc;

return
