
addpath(genpath('../../Modular_MatlabRoutines/'));

saveFolder = '~/Desktop/crossSectionPlots';

fNames1 = natsort(glob('/Volumes/Data/Cone_beam_results/Meeting_20190529_posterRecons/01_FDK_noJigCorr_paperFigs/03_slice_with_time/MSF/*.png'));
fNames2 = natsort(glob('/Volumes/Data/Cone_beam_results/Meeting_20190529_posterRecons/01_FDK_noJigCorr_paperFigs/03_slice_with_time/MRF/*.png'));
fNames3 = natsort(glob('/Volumes/Data/Cone_beam_results/Meeting_20190529_posterRecons/01_FDK_noJigCorr_paperFigs/03_slice_with_time/FBP/*.png'));

fNames = [fNames1(:) fNames2(:) fNames3(:)];

colors{1} = 'b';
colors{2} = 'r';
colors{3} = 'g';
colors{4} = 'c';
colors{5} = 'm';
colors{6} = 'k';

lines{1} = '-d';
lines{2} = '-*';
lines{3} = '-s';

imgNames{1} = 'Multi-Slice Fusion';
imgNames{2} = 'MBIR+4D-MRF';
imgNames{3} = 'FBP (3D)';


fullName = 'Compare Cross Section';
isCrop = false;
isNormalizeByFirst = true;
isreducedMargin = true;
isProfile = true;
magnification = 3;
LineWidth = 2;

params.saveFolder = saveFolder;
params.fNames = fNames;
params.imgNames = imgNames;
params.fullName = fullName;
params.isCrop = isCrop;
params.isNormalizeByFirst = isNormalizeByFirst;
params.isreducedMargin = isreducedMargin;
params.isProfile = isProfile;
params.magnification = magnification;
params.colors = colors;
params.lines = lines;
params.LineWidth = LineWidth;

params

createFolder_purge(saveFolder);

[profileImg_list, ~] = imageCompare_multiSlice(params);


% %%
% % figure, 
% % zerVal = 0.08;
% % plot(profile_list{3}-zerVal, 'g-s');
% % hold on;
% % plot(profile_list{2}+0.02-zerVal, 'r-d');
% % plot(profile_list{1}-zerVal, 'b-*');
% % legend(titles{3},titles{2},titles{1});



