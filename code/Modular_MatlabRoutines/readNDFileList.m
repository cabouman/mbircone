function [ fileList ] = readNDFileList(fileList_fName)


C = textread(fileList_fName, '%s','delimiter', '\n');

numDims = str2num(C{1});
sizes = str2num(C{2});
numFiles = prod(sizes);

fileList = C(3:end);

for i = 1:length(fileList)
    fileList(i) = absolutePath_relativeTo(fileList(i), fileList_fName);
end

if(numDims>1)
    fileList = reshape(fileList, sizes);
else
    % no action necessary.
    % here the array will be an Nx1 array
end




return


