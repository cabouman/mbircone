clearvars opts
folderName_binary = '/Volumes/ORANGE4TB_DataSets/Binaries/CBMBIR_workfolder/small-hemisphere-200-2e8/';

% clipLo = -0.2;
% clipHi = 1.2;
% radius = clipHi - clipLo;
% opts.target_lo = (0-clipLo)/radius;
% opts.target_hi = (1-clipLo)/radius;

opts.target_lo = 0.2/1.4;
opts.target_hi = 1.2/1.4;



%opts.mode = 'percentile 0.00 100';
opts.mode = 'percentile 0.5 99.9';
%opts.mode = 'absolute 0 10';


opts.prctileSubsampleFactor = 1;
%opts.mode = 'absolute -1 1';
opts.isGenerateMP4 = 1;
opts.isFramesJPG = 0;
opts.isFramesTIF = 1;
opts.isGenerateGIF = 1;
opts.isFramesPNG = 0;
opts.isGenerateColorbarEnvironment = 0;

opts.indexOrder = [1 2 3];
opts.flipVect = [0 0 0];



opts.figurePrintSize = 1;
opts.figureDispSize = 1;
opts.relFontSize = 1;

opts.range1 = '1:-5:end';
opts.range2 = '1:-5:end';
opts.range3 = '1:-1:end';

opts.folderSuffix = '_view';



step = 20;
count = 644;

mode = 0;

if(mode==1)
    % GE Stuff
    start = 62;
end
if(mode==2) 
    % MBIR Stuff
    start = 152;
end
if(mode>0)
    opts.range3 = [num2str(start), ':', num2str(step), ':', num2str(start+count-1) ]; 
end

view3D_folder( folderName_binary, opts);



% opts.indexOrder = [1 2 3]; 
% opts.folderSuffix = ['_view', sprintf('%d%d%d', opts.indexOrder(1), opts.indexOrder(2), opts.indexOrder(3)) ];
% view3D_folder( folderName_binary, opts);
% 
% opts.indexOrder = [1 3 2]; 
% opts.folderSuffix = ['_view', sprintf('%d%d%d', opts.indexOrder(1), opts.indexOrder(2), opts.indexOrder(3)) ];
% view3D_folder( folderName_binary, opts);
% 
% opts.indexOrder = [3 2 1]; 
% opts.folderSuffix = ['_view', sprintf('%d%d%d', opts.indexOrder(1), opts.indexOrder(2), opts.indexOrder(3)) ];
% view3D_folder( folderName_binary, opts);




