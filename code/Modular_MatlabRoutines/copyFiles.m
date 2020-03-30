function time = copyFiles( sourceList, destList )
% This copies all files from sourceList to destList
% note: sourceList can also be a singe string or a string cell array
% if sourceList is single item, then it gets copied to all the destList locations

sourceList = makeCell(sourceList);
destList = makeCell(destList);

for i = 1:length(destList)

	if(length(sourceList)==1)
		copyfile(sourceList{1}, destList{i});
	else
		copyfile(sourceList{i}, destList{i});
	end

end

end






