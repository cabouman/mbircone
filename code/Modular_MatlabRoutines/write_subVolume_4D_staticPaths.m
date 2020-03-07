function [] = write_subVolume_4D_staticPaths( fNameList_3D_fName, fName_timeVol )


fNameList_3D = readFileList(fNameList_3D_fName);

write_subVolume_4D( fNameList_3D, fName_timeVol )

return
