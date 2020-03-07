function [] = deadPixelCorrection( masterFile, plainParamsFile )

mfilepath=fileparts(which(mfilename));

addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));
addpath(fullfile(mfilepath,'../Preprocessing/Modular_PreprocessingRoutines'));

binaryFNames = readBinaryFNames( masterFile, plainParamsFile );
preprocessingParams = readPreprocessingParams(masterFile, plainParamsFile);
binaryFNames.origWght = prependAppendFileName( '', binaryFNames.wght, '.orig');



%%
errsino = read3D( binaryFNames.errsino, 'float32');
origWght = read3D( binaryFNames.origWght, 'float32');



%%
sumErr = sum((errsino), 3);

fName = prependAppendFileName( 'deadPixel/sumErr_', binaryFNames.sino, '');
write3D(fName, sumErr, 'float32');


%%
filtErr = sumErr - imgaussfilt(sumErr, 2.5);
filtErr = filtErr - medfilt2(filtErr, [3,3], 'symmetric');
filtErr = abs(filtErr);

fName = prependAppendFileName( 'deadPixel/filtErr_', binaryFNames.sino, '');
write3D(fName, filtErr, 'float32');


%%
k = 30;
[~, bin_arr]  = ksmallest(-filtErr, k); % k largest elements are binary 1

fName = prependAppendFileName( 'deadPixel/bin_arr_', binaryFNames.wght, '');
write3D(fName, bin_arr, 'float32');



%%
wght_new = zeros(size(origWght));
for i = 1:size(wght_new, 3)
	wght_new(:, :, i) = origWght(:, :, i) .* (1-bin_arr);
end



%%
write3D(binaryFNames.wght, wght_new, 'float32');






