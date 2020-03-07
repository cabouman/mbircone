function time = clear_BinaryFiles( control_4D , plainParamsFile )


tic

for t=1:control_4D.numVols_inversion
    system(['rm ' control_4D.binaryFnames_C.x_fwd{t} ]);
    system(['rm ' control_4D.binaryFnames_C.u_fwd{t} ]);
    system(['rm ' control_4D.binaryFnames_C.x_fwd_before{t} ]);
end

for t=1:control_4D.numVols_denoising
    for n=1:control_4D.num_decentral_denoisers
    	system(['rm ' control_4D.binaryFnames_C.x_dec_Prior{n,t} ]);
        system(['rm ' control_4D.binaryFnames_C.u_dec_Prior{n,t} ]);
    end
end

for t=1:control_4D.numVols_denoising
	system(['rm ' control_4D.binaryFnames_C.x_avg{t} ]);
    system(['rm ' control_4D.binaryFnames_C.u_avg{t} ]);
    system(['rm ' control_4D.binaryFnames_C.x_cent_prior{t} ]);
end


time = toc;

return
