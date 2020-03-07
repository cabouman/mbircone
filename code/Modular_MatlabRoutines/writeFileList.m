function [ ] = writeFileList(fileList_fName, fileList)

numFiles = length(fileList);

fileID = fopen(fileList_fName, 'w');
fprintf(fileID,'%d\n', numFiles);

for i=1:numFiles
	fprintf(fileID,'%s\n', fileList{i});
end

fprintf(fileID,'%\n\n');
fclose(fileID);

return