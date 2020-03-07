function time = consensus_compute_avg( avgVol_fNameList, fwdVol_fNameList, priorVol_fNameList, consensus_mode, numVols_inversion, num_decentral_denoisers, averaging_wtList )

tic

if size(priorVol_fNameList,1)+1 ~= length(averaging_wtList)
	error('consensus_compute_avg: size mismatch');
end

switch consensus_mode
case 'time'

	for t=1:numVols_inversion

		vol = averaging_wtList(1)*read3D( fwdVol_fNameList{t} , 'float32' );
		normalizeFactor = averaging_wtList(1);
		for n=1:num_decentral_denoisers
			vol = vol + averaging_wtList(n+1)*read3D( priorVol_fNameList{n,t} , 'float32' );
			normalizeFactor = normalizeFactor + averaging_wtList(n+1);
		end

		vol = vol / normalizeFactor;
		write3D( avgVol_fNameList{t}, vol, 'float32');

	end

case 'view'

	vol = 0 * read3D( fwdVol_fNameList{1} , 'float32' );
	normalizeFactor = 0;
	for t=1:numVols_inversion
	    vol = vol + averaging_wtList(1)*read3D( fwdVol_fNameList{t} , 'float32' );
	    normalizeFactor = normalizeFactor + averaging_wtList(1);
	end
	for n=1:num_decentral_denoisers
		vol = vol + averaging_wtList(n+1)*read3D( priorVol_fNameList{n,1} , 'float32' );
		normalizeFactor = normalizeFactor + averaging_wtList(n+1);
	end

	vol = vol / normalizeFactor;
	write3D( avgVol_fNameList{1}, vol, 'float32');

end  


time = toc;