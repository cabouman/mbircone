
addpath(genpath('../../Modular_MatlabRoutines/'));

fNameList = natsort(glob('/Volumes/Data/Cone_beam_results/RenderResults/*.recon'));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Default Params
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts1.permuteSlice = [1 2];
opts1.flipSlice = [0 0];
opts1.sliceDim = 1;
opts1.sliceId = 380;
opts1.folderSuffix = 'view_4D';
opts1.name = 'slicing';

opts2.permuteSlice = [1 2];
opts2.flipSlice = [0 0];
opts2.sliceDim = 2;
opts2.sliceId = 380;
opts2.folderSuffix = 'view_4D';
opts2.name = 'slicing';

opts3.permuteSlice = [1 2];
opts3.flipSlice = [0 0];
opts3.sliceDim = 3;
opts3.sliceId = -1;
opts3.folderSuffix = 'view_4D';
opts3.name = 'slicing';

Convert_4D_to_3D(fNameList, {opts1, opts2, opts3});

