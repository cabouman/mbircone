clearvars opts
folderName_binary = '/Users/tbalke/Desktop/nsi';

opts.target_lo = 0.2/1.4;
opts.target_hi = 1.2/1.4;
%opts.mode = 'percentile 0.00 100';
opts.mode = 'percentile 0.01 99.99';

%opts.mode = 'absolute -0.2 1.2';
%opts.mode = 'absolute 0 1';

opts.prctileSubsampleFactor = 10;
%opts.mode = 'absolute -1 1';
opts.isGenerateMP4 = 1;
opts.isFramesJPG = 0;
opts.isFramesTIF = 1;
opts.isGenerateGIF = 1;
opts.isGenerateColorbarEnvironment = 0;

%opts.indexOrder = [3 2 1]; 
opts.flip = 0;
opts.rotate_90 = 0;


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




