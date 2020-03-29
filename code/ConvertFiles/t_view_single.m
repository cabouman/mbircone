clearvars opts
fName = '/Volumes/DataSets/Binaries/CBMBIR_workfolder/Al_rod.recon';



opts.mode = 'absolute 0 0.05';
%opts.mode = 'percentile 1 99';
opts.prctileSubsampleFactor = 10;
opts.isGenerateMP4 = 1;
opts.isFramesJPG = 1;
opts.isFramesTIF = 1;

%opts.indexOrder = [1 2 3]; 
opts.flip = 0;
opts.rotate_90 = 0;

opts.figurePrintSize = 2;
opts.figureDispSize = 2;
opts.relFontSize = 1;

opts.range1 = 'end/3:2*end/3';
opts.range2 = 'end/3:2*end/3';
opts.range3 = 'end/3:end/3+3';

opts.folderSuffix = '_view';

view3D_single( fName, opts);

