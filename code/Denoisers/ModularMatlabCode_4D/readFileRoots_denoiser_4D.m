function [ fRoots ] = readFileRoots_denoiser_4D( masterFile, plainParamsFile )

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'binaryFNames';
value = '';


fRoots.noisyImageFNameRoot = plainParams(executablePath, get_set, masterFile, 'noisyImageFNameRoot', '', value, '-r');
fRoots.denoisedImageFNameRoot = plainParams(executablePath, get_set, masterFile, 'denoisedImageFNameRoot', '', value, '-r');

fRoots.noisyBinaryFName_timeList = plainParams(executablePath, get_set, masterFile, 'noisyBinaryFName_timeList', '', value, '-r');
fRoots.denoisedBinaryFName_timeList = plainParams(executablePath, get_set, masterFile, 'denoisedBinaryFName_timeList', '', value, '-r');

return
