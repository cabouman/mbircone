saveFolder = '/Users/tbalke/Desktop/_figs/easy_crop';

fNames = {};
i = 0;

i = i+1; fNames{i} = '/Users/tbalke/Dropbox/Shared/share_soumendu/deeplearningproject/Report/Figs/recons/easy/TEST_easy.recon0042.tif'; 
i = i+1; fNames{i} = '/Users/tbalke/Dropbox/Shared/share_soumendu/deeplearningproject/Report/Figs/recons/easy/TEST_easy.noisy.recon0042.tif'; 
i = i+1; fNames{i} = '/Users/tbalke/Dropbox/Shared/share_soumendu/deeplearningproject/Report/Figs/recons/easy/TEST_easy.noisy.denoised_bm4d.recon0042.tif'; 
i = i+1; fNames{i} = '/Users/tbalke/Dropbox/Shared/share_soumendu/deeplearningproject/Report/Figs/recons/easy/TEST_easy.noisy.denoised_bm4d_diff.recon0042.tif'; 
i = i+1; fNames{i} = '/Users/tbalke/Dropbox/Shared/share_soumendu/deeplearningproject/Report/Figs/recons/easy/TEST_easy.noisy_diff.recon0042.tif'; 
i = i+1; fNames{i} = '/Users/tbalke/Dropbox/Shared/share_soumendu/deeplearningproject/Report/Figs/recons/easy/TEST_easy.denoised_expt_3.recon0042.tif'; 
i = i+1; fNames{i} = '/Users/tbalke/Dropbox/Shared/share_soumendu/deeplearningproject/Report/Figs/recons/easy/TEST_easy.denoised_expt_3_diff.recon0042.tif'; 
i = i+1; fNames{i} = '/Users/tbalke/Dropbox/Shared/share_soumendu/deeplearningproject/Report/Figs/recons/easy/TEST_easy.denoised_expt_2.recon0042.tif'; 
i = i+1; fNames{i} = '/Users/tbalke/Dropbox/Shared/share_soumendu/deeplearningproject/Report/Figs/recons/easy/TEST_easy.denoised_expt_2_diff.recon0042.tif'; 
i = i+1; fNames{i} = '/Users/tbalke/Dropbox/Shared/share_soumendu/deeplearningproject/Report/Figs/recons/easy/TEST_easy.denoised_expt_1.recon0042.tif'; 
i = i+1; fNames{i} = '/Users/tbalke/Dropbox/Shared/share_soumendu/deeplearningproject/Report/Figs/recons/easy/TEST_easy.denoised_expt_1_diff.recon0042.tif'; 
i = i+1; fNames{i} = '/Users/tbalke/Dropbox/Shared/share_soumendu/deeplearningproject/Report/Figs/recons/easy/TEST_easy.denoised_expt_0.recon0042.tif'; 
i = i+1; fNames{i} = '/Users/tbalke/Dropbox/Shared/share_soumendu/deeplearningproject/Report/Figs/recons/easy/TEST_easy.denoised_expt_0_diff.recon0042.tif'; 
i = i+1; fNames{i} = '/Users/tbalke/Dropbox/Shared/share_soumendu/deeplearningproject/Report/Figs/recons/easy/TEST_easy_diff.recon0042.tif'; 

colors = {};
titles = {};
lines = {};
for i = 1:length(fNames)
    [~, titles{i}] = fileparts(fNames{i});
    colors{i} = 'r';
    lines{i} = '-';
end

colors{1} = 'b';
colors{2} = 'r';
colors{3} = 'g';
colors{4} = 'c';
colors{5} = 'm';
colors{6} = 'k';

lines{1} = '-d';
lines{2} = '-*';
lines{3} = '-s';

[~, titles{1}] = fileparts(fNames{1});
[~, titles{2}] = fileparts(fNames{2});
[~, titles{3}] = fileparts(fNames{3});


titles_all = 'CoCrHoles_315Views_QGGMRF';

magnification = 3;



xi = [10, 20];
yi = [100, 110];

isCrop = true;
isreducedMargin = false;
isProfile = true;

profile_list = imageCompare(saveFolder, fNames, titles, titles_all, isProfile, xi, yi, colors, lines, isCrop, magnification, isreducedMargin);
