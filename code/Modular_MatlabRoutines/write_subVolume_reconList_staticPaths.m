function [] = write_subVolume_reconList_staticPaths( fNameList_3D_fName, fName_timeVol )


fNameList_3D = readFileList(fNameList_3D_fName);

write_subVolume_reconList( fNameList_3D, fName_timeVol );

return
