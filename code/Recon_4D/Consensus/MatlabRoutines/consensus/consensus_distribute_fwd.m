function time = consensus_distribute_fwd( dest_fwdVol_fNameList, src_centVol_fNameList, src_fwdVol_fNameList, coeff_cent, coeff_fwd, consensus_mode, numVols_inversion )

tic

for t=1:numVols_inversion

    switch consensus_mode
    case 'time'
        vol_src_cent = read3D( src_centVol_fNameList{t}, 'float32');
    case 'view'
        vol_src_cent = read3D( src_centVol_fNameList{1}, 'float32');
    end

    vol_src_fwd = read3D( src_fwdVol_fNameList{t}, 'float32');

    vol_dest = coeff_cent*vol_src_cent + coeff_fwd*vol_src_fwd ;

    write3D( dest_fwdVol_fNameList{t}, vol_dest , 'float32');
end

time = toc;

return
