
addpath(genpath('../../Modular_MatlabRoutines/'))

fNameList_in = glob('/Volumes/Data/Cone_beam_results/2018/Meeting_20181121/01_4D_recons/02_invWt=2/05_CE_CNN_all_again/CE/*.recon');
dir_out = '/Volumes/Data/Cone_beam_results/RenderResults/';

opts.offset = [20,20,0];
opts.limits_lo = [81 106 17]-opts.offset;
opts.limits_hi = [534 562 87]+opts.offset;
opts.description = 'crop';

process_reconList( fNameList_in, opts, dir_out );
