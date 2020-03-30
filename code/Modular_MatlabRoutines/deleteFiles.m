function deleteFiles( fileList )
% This deletes all files from fileList
% note: fileList can also be a singe string or a string cell array

fileList = makeCell(fileList);

for i = 1:length(fileList)

	delete(fileList{i});

end

end






