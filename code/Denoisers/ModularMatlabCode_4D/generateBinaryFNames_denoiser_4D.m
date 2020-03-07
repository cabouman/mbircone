function [ noisyFname_list, denoisedFname_list ] = generateBinaryFNames_denoiser_4D( masterFile, plainParamsFile, numTimes, Prefix, Suffix )

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines/'));

[ fRoots ] = readFileRoots_denoiser_4D( masterFile, plainParamsFile );

noisyFname_list = generate_4DfNamesList( fRoots.noisyImageFNameRoot, Prefix, Suffix, 1, numTimes );
writeFileList( fRoots.noisyBinaryFName_timeList, noisyFname_list );

denoisedFname_list = generate_4DfNamesList( fRoots.denoisedImageFNameRoot, Prefix, Suffix, 1, numTimes);
writeFileList( fRoots.denoisedBinaryFName_timeList, denoisedFname_list );




return
