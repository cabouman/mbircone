
addpath(genpath('../../Modular_MatlabRoutines/'));

fNameList = natsort(glob('../../../**/object.recon'))

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

volRender_reconList(fNameList, opts);
%surfRender_reconList(fNameList, opts);
