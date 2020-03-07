function [hyperVol_fname] = write_4D_visualization( fNameList_3D, dirNameRoot, fNameRoot, coord )


recon_Fname = fNameList_3D{1};
[recon_dir, recon_nakedName, recon_ext] = fileparts(recon_Fname);

hyperVol_dir = [ recon_dir '/' dirNameRoot ];
hyperVol_fname = [ hyperVol_dir '/' fNameRoot recon_ext ];

if exist(hyperVol_dir)==0
	mkdir(hyperVol_dir)
end

write_subVolume_4D( fNameList_3D, hyperVol_fname, coord );

return
