saveFolder = '/Volumes/Data/Cone_beam_results/Meeting_20181128/01_figures_paper/yzView_rightSection_016_crossSection';

fNames = {};
fNames{1} = '/Volumes/Data/Cone_beam_results/Meeting_20181128/01_figures_paper/yzView_rightSection_016_crossSection/qggmrf_4D.tif';
fNames{2} = '/Volumes/Data/Cone_beam_results/Meeting_20181128/01_figures_paper/yzView_rightSection_016_crossSection/all.tif';

titles = {};
% [~, titles{1}] = fileparts(fNames{1});
% [~, titles{2}] = fileparts(fNames{2});
titles{1} = 'qggmrf 4D';
titles{2} = 'our';

titles_all = 'Compare Cross Section';

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
isreducedMargin = true;

imageCompare(saveFolder, fNames, titles, titles_all, isProfile, xi, yi, colors, lines, isCrop, magnification, isreducedMargin);

