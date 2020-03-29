


fileList_in = glob('/Volumes/Data/Cone_beam_results/Meeting_20190102/04_video/03_cropped2/*.recon');
fileList_in = {'/Volumes/Data/Cone_beam_results/Meeting_20190102/04_video/03_cropped2/01_vol_qggmrf_crop.recon', '/Volumes/Data/Cone_beam_results/Meeting_20190102/04_video/03_cropped2/02_vol_CNN_CE_crop.recon'}
catAxis = 1;
padLen = 20;
padIntensity = 100;
fName_out = '/Volumes/Data/Cone_beam_results/RenderResults/joined.recon';

concat_recon( fileList_in, catAxis, padLen, padIntensity, fName_out );
