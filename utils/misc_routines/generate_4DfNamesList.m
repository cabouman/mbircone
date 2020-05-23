function [ fName_list ] = generate_4DfNamesList(fNameRoot, Prefix, Suffix, i_start, i_stop, i_step)

if exist('i_step', 'var')==0
	i_step = 1;
end

id_list = i_start:i_step:i_stop;
N = length(id_list);
fName_list = cell(1,N);

[filePath, nakedName, ext] = fileparts(fNameRoot);
numDigits = 6;
formatstring = ['%0', num2str(numDigits), 'd'];
for i = 1:N
	id = id_list(i);
	fName_list{i} = [filePath, '/', Prefix, num2str(id, formatstring), '_', nakedName, Suffix, ext];
end



return