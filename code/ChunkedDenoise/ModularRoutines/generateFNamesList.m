function [ fName_list ] = generateFNamesList(folderSuffix, fName, i_start, i_stop)

N = i_stop - i_start + 1;

[filePath, nakedName, ext] = fileparts(fName);
numDigits = 6;
formatstring = ['%0', num2str(numDigits), 'd'];
for i = i_start:i_stop
	fName_list{i-i_start+1} = [filePath, '/', nakedName, folderSuffix, '/', nakedName, '_', num2str(i, formatstring), ext];
end



end