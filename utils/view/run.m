clearvars opts

folderName_binary = '../../demo/inversion/';

opts.target_lo = 0;
opts.target_hi = 1;

% opts.mode = 'absolute 0 0.1';
opts.mode = 'percentile 1 99';

opts.prctileSubsampleFactor = 10;

opts.isGenerateMP4 = 1;
opts.isFramesJPG = 0;
opts.isFramesTIF = 1;
opts.isGenerateGIF = 1;
opts.isFramesPNG = 1;
opts.isGenerateColorbarEnvironment = 0;

opts.indexOrder = [1 2 3];


% small disp: 2,1,1
% large disp: 2,2,1 ?
opts.figurePrintSize = 2;
opts.figureDispSize = 1;
opts.relFontSize = 1;


% none
opts.range1 = '1:end';
opts.range2 = '1:end';
opts.range3 = '2:1';

opts.folderSuffix = '_view';

% % 
opts.indexOrder = [1 2 3];
opts.folderSuffix = '_view[1 2 3]';
opts.flipVect = [0 0 0];
% % opts.range1 = '370:464';
% % opts.range2 = '182:320';
% % opts.range3 = '18:19';
view3D_folder( folderName_binary, opts);


% opts.indexOrder = [3 1 2];
% opts.folderSuffix = '_view[3 1 2]';
% opts.flipVect = [1 0 0];
% opts.range1 = '1:end';
% opts.range2 = '1:end';
% opts.range3 = '2:1';
% view3D_folder( folderName_binary, opts);

close all
