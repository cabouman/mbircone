function [imgStack] = readImgStack(stackDir)

fNameList = natsort(glob([stackDir, '/*']));

for i=1:length(fNameList)

	img = imread(fNameList{i});
	maxVal = intmax(class(img));
	img = double(img)/double(maxVal);
	imgList{i} = img;

end

imgStack = zeros(size(imgList{1},1), size(imgList{1},2), length(fNameList));

for i=1:length(fNameList)

	imgStack(:,:,i) = imgList{i};

end

return