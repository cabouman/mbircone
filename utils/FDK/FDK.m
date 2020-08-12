function [ ] = FDK(masterFile, plainParamsFile)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../Preprocessing/Modular_PreprocessingRoutines'));
addpath(fullfile(mfilepath,'../misc_routines_matlab'));
addpath('CBCT_Kyungsang_matlab_Feb2015');
addpath('CBCT_Kyungsang_matlab_Feb2015/bin');


% MBIR Params
binaryFNames = readBinaryFNames( masterFile, plainParamsFile );
sinoParams = readSinoParams( masterFile, plainParamsFile );
imgParams = readImgParams( masterFile, plainParamsFile );
angleList = readViewAngleList( masterFile, plainParamsFile );

% Converted Params
paramSetting_MBIR;


proj = read3D(binaryFNames.sino, 'float32');
proj = permute(proj, [2 1 3]);

%% Recon case 1 - Analytic reconstruction: filtered backprojection
% filter='ram-lak','shepp-logan','cosine', 'hamming', 'hann' : (ramp + additional filter)
param.filter='hann'; 

proj_filtered = filtering(proj,param);
Reconimg = CTbackprojection(proj_filtered, param);

Reconimg = permute(Reconimg, [3 1 2]);

disp('FDK done')

% write3D('out/proj_filtered.sino', proj_filtered, 'float32');
write3D(binaryFNames.recon, Reconimg, 'float32');

end