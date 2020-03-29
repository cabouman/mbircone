
addpath(genpath('../../Modular_MatlabRoutines/'));
addpath(genpath('../MatlabRoutines/'));

saveFolder = '~/Desktop/crossSectionPlots';

fNames = {};
i=0;
i = i+1; fNames{i} = '/Users/smajee/Dropbox/Temp/phantom_4D_k=0_t=4_object.phantom.recon0005.png';
i = i+1; fNames{i} = '/Users/smajee/Dropbox/Temp/FBP_4D_k=0_t=4_object.recon0005.png';
i = i+1; fNames{i} = '/Users/smajee/Dropbox/Temp/MRF4D_z_cent_prior_t=5.recon0005.png';
i = i+1; fNames{i} = '/Users/smajee/Dropbox/Temp/MSF_z_cent_prior_t=5.recon0005.png';

imgNames = {};
i=0;
i = i+1; imgNames{i} = 'Phantom'
i = i+1; imgNames{i} = 'FBP (3D)';
i = i+1; imgNames{i} = 'MBIR+4D-MRF';
i = i+1; imgNames{i} = 'Multi-Slice Fusion';

colors{1} = 'k';
colors{2} = 'g';
colors{3} = 'r';
colors{4} = 'b';
colors{5} = 'm';
colors{6} = 'k';

lines{1} = '-d';
lines{2} = '-*';
lines{3} = '-s';
lines{4} = '-o';

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

[profile_list, ~] = imageCompare(params);




