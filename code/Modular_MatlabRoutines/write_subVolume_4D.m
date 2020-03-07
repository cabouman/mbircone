function [] = write_subVolume_4D( fNameList_3D, fName_timeVol, coord )

timVol = [];
for t=1:length(fNameList_3D)
	recon_Fname = fNameList_3D{t};
	x = read3D( recon_Fname, 'float32');
	x = permute(x , [3 2 1]);
	timVol(:,:,t) = subSelectionVol(x, coord);
end
timVol = permute(timVol, [3 2 1]);

write3D( fName_timeVol, timVol, 'float32');

return


function img = subSelectionVol(vol_orig, coord)

N_coord = size(vol_orig, coord);
i_coord = round(N_coord/2);

switch coord
case 1
	img = vol_orig(i_coord,:,:);
	img  = shiftdim(img);
	img = rot90(img, 1);

case 2 
	img = vol_orig(:,i_coord,:);
	img  = permute(img, [1 3 2]);
	img = rot90(img, 1);

case 3
	img = vol_orig(:,:,i_coord);

otherwise
	error('subSelectionVol: wrong coord value')
end


return
