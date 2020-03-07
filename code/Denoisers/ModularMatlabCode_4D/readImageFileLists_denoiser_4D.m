function [ fileLists ] = readImageFileLists_denoiser_4D( masterFile, plainParamsFile )


[ fRoots ] = readFileRoots_denoiser_4D( masterFile, plainParamsFile );

fileLists.noisyImageNames = readFileList(fRoots.noisyBinaryFName_timeList);
fileLists.denoisedImageNames = readFileList(fRoots.denoisedBinaryFName_timeList);


return
