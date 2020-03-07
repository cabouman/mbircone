function [ fileList ] = readFileList(fileList_fName)


C = textread(fileList_fName, '%s','delimiter', '\n');
numFiles = str2num(C{1});
fileList = cell(1, numFiles);


% Read individual filenames
for i=1:(numFiles)	
	% Read individual 3D vols 
	fName_rel = C{i+1};
	fileList{i} = absolutePath_relativeTo(fName_rel, fileList_fName);
end


return