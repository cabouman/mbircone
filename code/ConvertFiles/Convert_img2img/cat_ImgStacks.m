function [img_cat] = cat_ImgStacks(ImgStackList_in, ImgStack_out, params)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));
addpath(fullfile(mfilepath,'../MatlabRoutines'));

imgStackList = {};
for i=1:length(ImgStackList_in)

	stackDir = ImgStackList_in{i};
	imgStackList{i} = readImgStack(stackDir);

end

img_cat = imgStackList{1};

padLens = [size(img_cat,1), size(img_cat,2), size(img_cat,3)];
padLens(params.catAxis) = params.padLen;
padImg = params.padIntensity * ones( padLens );


for i=2:length(imgStackList)

	img = imgStackList{i};
	img_cat = cat(params.catAxis, img_cat, padImg, img);

end


createFolder_purge(ImgStack_out);

generateGif( ImgStack_out, params.dispName, img_cat );
generateMPEG4Video( ImgStack_out, params.dispName, img_cat );
generateSingleImages( ImgStack_out, params.dispName, img_cat, 'tif');
generateSingleImages( ImgStack_out, params.dispName, img_cat, 'png');

