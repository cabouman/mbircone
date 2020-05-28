function [ img ] = read3D( fName, fileDataType, dataTypeOut)
% fileDataType: datatype of binary file data
% dataTypeOut:  datatype of the variable called 'img' ('double', 'single',
% 'logical','int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64')

if ~exist('dataTypeOut','var') || isempty('dataTypeOut')
  dataTypeOut='single';
end

fid = fopen(fName, 'r');

if(fid == -1)
	error(['Cannot open file ', fName]);
end

% size(1): slowest index, ..., size(3): fastest index
sizes = (fread(fid, 3, 'int32'))';


img = zeros(sizes(3), sizes(2), sizes(1), dataTypeOut);
for i=1:sizes(1)
    temp = fread(fid, sizes(3)*sizes(2), fileDataType);
    img(:,:,i) = reshape(temp, [sizes(3), sizes(2)]);
end

numNaNs = sum(isnan(img(:)));
if(numNaNs>0)
    warnMess = ['Number of NaNs in "', fName, '" = ', num2str(numNaNs)];
    warning(warnMess);
end

fclose(fid);

end


% 
% function [ img ] = read3D( fName, dataType )
% 
% fid = fopen(fName, 'r');
% 
% if(fid == -1)
% 	error(['Cannot open file ', fName]);
% end
% 
% sizes = fread(fid, 3, 'int32');
% 
% %flip sizes to match C notation (row major)
% sizes = sizes(end:-1:1);
% 
% fullsize = sizes(1)*sizes(2)*sizes(3);
% 
% img = reshape(fread(fid, fullsize, dataType), sizes');
% 
% fclose(fid);
% 
% end
% 
