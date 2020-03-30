
addpath(genpath('../../Modular_MatlabRoutines/'));

fNameList = natsort(glob('/Volumes/Data/Cone_beam_results/RenderResults/*.recon'))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Default Params
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts.mode = 'percentile 1 99';
opts.prctileSubsampleFactor = 10;
opts.indexOrder = [1 2 3];
opts.flipVect = [0 0 0];
% opts.transparencyLims = [0 1];
opts.range1 = '1:end';
opts.range2 = '1:end';
opts.range3 = '1:end';

% volume render specific options
opts.ScaleFactors = [1, 1, 1];
opts.CameraPosition = [1.0 0 0.32];
opts.CameraTarget = [0 0 0];
opts.Isovalue = 0.5;
opts.Renderer = 'VolumeRendering'; % VolumeRendering, MaximumIntensityProjection, Isosurface
opts.surfaceColor = 'w';
opts.BackgroundColor = 'k';

opts.folderSuffix = 'view_4D';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dataset Specific Params
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% kwikpen-static
% opts.mode = 'absolute 0.05 0.3';
% opts.range1 = '215:385';
% opts.range2 = '135:305';
% opts.range3 = '180:315';

% kwikpen-dynamic
% opts.mode = 'absolute 0.05 0.3';
% opts.range1 = '90:200';
% opts.range2 = '90:200';
% opts.range3 = '75:190';

% vial
% opts.mode = 'absolute 0 0.1';
% opts.flipVect = [0 0 0];
% opts.transparencyLims = [0.05 0.1];
% % opts.range1 = '60:540';
% % opts.range2 = '70:580';
% % opts.range3 = '1:end';
% opts.CameraPosition = [0.0 1.0 0.32];

% reverse eng
opts.Isovalue = 0.2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% volRender_reconList(fNameList, opts);
surfRender_reconList(fNameList, opts);
