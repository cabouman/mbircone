function binaryFnames_C = CADMM_genBinaryFileNames( control_4D ) 


binaryFnames_C.x_fwd = cell(1,control_4D.numVols_inversion);
for t=1:control_4D.numVols_inversion
	binaryFnames_C.x_fwd{t} = genBinaryFileName_single( control_4D.binaryFroots_C.x_fwd, control_4D.binaryFroots_C.fwd_suffix, t);
end
binaryFnames_C.u_fwd = cell(1,control_4D.numVols_inversion);
for t=1:control_4D.numVols_inversion
    binaryFnames_C.u_fwd{t} = genBinaryFileName_single( control_4D.binaryFroots_C.u_fwd, control_4D.binaryFroots_C.fwd_suffix, t);
end

binaryFnames_C.x_dec_Prior = cell(control_4D.num_decentral_denoisers, control_4D.numVols_denoising);
for n=1:control_4D.num_decentral_denoisers
    for t=1:control_4D.numVols_denoising
        temp_name = genBinaryFileName_single( control_4D.binaryFroots_C.x_dec_Prior, control_4D.binaryFroots_C.dec_Prior_suffix, n);
        temp_name2 = genBinaryFileName_single( temp_name, control_4D.binaryFroots_C.fwd_suffix, t);

        binaryFnames_C.x_dec_Prior{n,t} = temp_name2;
    end
end
binaryFnames_C.u_dec_Prior = cell(control_4D.num_decentral_denoisers, control_4D.numVols_denoising);
for n=1:control_4D.num_decentral_denoisers
    for t=1:control_4D.numVols_denoising
        temp_name = genBinaryFileName_single( control_4D.binaryFroots_C.u_dec_Prior, control_4D.binaryFroots_C.dec_Prior_suffix, n);
        temp_name2 = genBinaryFileName_single( temp_name, control_4D.binaryFroots_C.fwd_suffix, t);

        binaryFnames_C.u_dec_Prior{n,t} = temp_name2;
    end
end



binaryFnames_C.x_avg = cell(1,control_4D.numVols_denoising);
for t=1:control_4D.numVols_denoising
    binaryFnames_C.x_avg{t} = genBinaryFileName_single( control_4D.binaryFroots_C.x_avg, control_4D.binaryFroots_C.fwd_suffix, t);
end

binaryFnames_C.u_avg = cell(1,control_4D.numVols_denoising);
for t=1:control_4D.numVols_denoising
    binaryFnames_C.u_avg{t} = genBinaryFileName_single( control_4D.binaryFroots_C.u_avg, control_4D.binaryFroots_C.fwd_suffix, t);
end



binaryFnames_C.x_cent_prior = cell(1,control_4D.numVols_denoising);
for t=1:control_4D.numVols_denoising
    binaryFnames_C.x_cent_prior{t} = genBinaryFileName_single( control_4D.binaryFroots_C.x_cent_prior, control_4D.binaryFroots_C.fwd_suffix, t);
end


binaryFnames_C.x_fwd_before = cell(1,control_4D.numVols_inversion);
for t=1:control_4D.numVols_inversion
    binaryFnames_C.x_fwd_before{t} = genBinaryFileName_single( control_4D.binaryFroots_C.x_fwd_before, control_4D.binaryFroots_C.fwd_suffix, t);
end


return

