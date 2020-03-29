
addpath(genpath('../../Modular_MatlabRoutines/'))

fNameList_in = glob('/Volumes/Data/Cone_beam_results/RenderResults/temp/*.recon');
dir_out = '/Volumes/Data/Cone_beam_results/RenderResults/';

% normalize_recon_darkBrightPatch( fNameList_in, dir_out, opts );

opts.dark = 0.0;
opts.bright = 0.05;
opts.darkbright_from_GUI = 1;
% opts.sliceID = 48;  % comment if undesired: will be taken as mid slice
opts.description = 'normalize';

process_reconList( fNameList_in, opts, dir_out );
