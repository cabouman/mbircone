function [ images ] = readImagesInIndexList( fNames, indexList )

numRequestedImages = length(indexList);

temp = double(imread(fNames{0+1}));

images = zeros(size(temp,1), size(temp,2), numRequestedImages);

for i_images = 1:numRequestedImages
	i_fNames = indexList(i_images);

	images(:,:,i_images) = double(imread(fNames{i_fNames+1}));

end




end

