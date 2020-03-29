
addpath(genpath('../../Modular_MatlabRoutines/'))

fNameList_in = glob('/Volumes/Data/Cone_Beam_Dataset/Lily_data/3D_scans/reverseengineer_device_scan20191104145822/recons/02_BHC/recon/*.recon');
dir_out = '/Volumes/Data/Cone_beam_results/RenderResults/';

opts.indexOrder = [3, 2, 1];
opts.flipVect = [0 1 0];
opts.description = 'orient';

process_reconList( fNameList_in, opts, dir_out );
