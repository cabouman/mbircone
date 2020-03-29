saveFolder = '/Users/tbalke/Desktop/save';

fNames = {};
fNames{1} = '/Volumes/ORANGE4TB_DataSets/MAI-recons/Al/1000Views/slices/Al_1000Views_BM4D_NOBHCSC.normalized.recon0044.tif';
fNames{2} = '/Volumes/ORANGE4TB_DataSets/MAI-recons/Al/1000Views/slices/Al_1000Views_QGGMRF_NOBHCSC.normalized.recon0044.tif';
fNames{3} = '/Volumes/ORANGE4TB_DataSets/MAI-recons/Al/1000Views/slices/NG8-GE-2-10_ScattCor.normalized.recon0464.tif';

titles = {};
[~, titles{1}] = fileparts(fNames{1});
[~, titles{2}] = fileparts(fNames{2});
[~, titles{3}] = fileparts(fNames{3});

titles_all = 'CoCr_1000Views_BM4D';

magnification = 3;

colors = {};
colors{1} = 'b';
colors{2} = 'r';
colors{3} = 'g';
colors{4} = 'c';
colors{5} = 'm';
colors{6} = 'k';


lines = {};
lines{1} = '-d';
lines{2} = '-*';

xi = [10, 20];
yi = [100, 110];

isProfile = true;
isCrop = true;
isreducedMargin = false;

imageCompare(saveFolder, fNames, titles, titles_all, isProfile, xi, yi, colors, lines, isCrop, magnification, isreducedMargin);
