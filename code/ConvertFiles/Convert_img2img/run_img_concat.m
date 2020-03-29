
dirName = '/Volumes/Data/Cone_beam_results/Meeting_20190529_posterRecons/01_FDK_noJigCorr_paperFigs/04_4D_cropped/'
ImgStackList_in = {[dirName, 'FBP'], [dirName, 'MRF'], [dirName, 'MSF']};
ImgStack_out = [dirName, 'cat'];

% ImgStackList_in = {'/Volumes/Data/Cone_beam_results/Meeting_20190508/03_new_vid_xt_vial/02_crop_norm/MRF', '/Volumes/Data/Cone_beam_results/Meeting_20190508/03_new_vid_xt_vial/02_crop_norm/MSF'}
% ImgStack_out = '/Volumes/Data/Cone_beam_results/Meeting_20190508/03_new_vid_xt_vial/03_catenated'

params.dispName = 'cat';
params.catAxis = 1;
params.padLen = 0; 
params.padIntensity = 0;

img_cat = cat_ImgStacks(ImgStackList_in, ImgStack_out, params);
