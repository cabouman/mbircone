function [img] = read4D(fNameList, dataType, dataTypeOut)

if ~exist('dataTypeOut','var') || isempty('dataTypeOut')
  dataTypeOut='single';
end

% read a 3D volume for each time point
for i=1:length(fNameList)
	temp_img = read3D(fNameList{i}, dataType, dataTypeOut);
	img(i,:,:,:) = temp_img;
end
