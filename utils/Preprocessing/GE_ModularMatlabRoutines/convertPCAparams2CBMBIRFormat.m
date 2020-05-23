function [ params ] = convertPCAparams2CBMBIRFormat( pcaData, preprocessingParams, viewIndexList)


% FOD: source - center: Focus - object distance
% FDD: source - detector: Focus - detector distance
params.u_s = -pcaData.FOD;
params.u_d0 = pcaData.FDD - pcaData.FOD;
M = (params.u_d0 - params.u_s)/ (-params.u_s);


% Assume rotation axis is on u-axis
params.u_r = 0; % should always be 0
params.v_r = 0; % adjust this to increase / decrease wobble

% Not sure, but assume
% X <-> v
% Y <-> w
params.Delta_dv = pcaData.PixelsizeX;
params.Delta_dw = pcaData.PixelsizeY;
params.N_dv = pcaData.DimX;
params.N_dw = pcaData.DimY;

% Assume cx and cy are the correspond to (u, v, w) = (u_d0, 0, 0).
% cx, cy are in units of detector width.
params.v_d0 = - pcaData.cx * pcaData.PixelsizeX + M*params.v_r;
params.w_d0 = - pcaData.cy * pcaData.PixelsizeY;



params.N_beta = length(viewIndexList);
params.TotalAngle = pcaData.RotationSector;
params.viewAngleList = computeAngleList( viewIndexList, pcaData.NumberImages, pcaData.RotationSector, preprocessingParams.rotationDirection);



end

