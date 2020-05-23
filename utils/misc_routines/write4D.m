function [] = write4D(fNameList, img, dataType)

if length(fNameList) ~= size(img,1)
	error('write4D: time dimension mismatch');
else
	% write a 3D volume for each time point
	for i=1:length(fNameList)
		temp_img = img(i,:,:,:);
		temp_img = shiftdim(temp_img,1);

		% disp(size(temp_img))
		write3D(fNameList{i}, temp_img, dataType);
	end
end