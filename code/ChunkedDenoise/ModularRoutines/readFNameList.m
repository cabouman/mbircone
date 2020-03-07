function [ fName_list ] = readFNameList(fName)


fid = fopen(fName, 'r');

N = fscanf(fid, '%d\n', [1,1]);
fName_list = textscan(fid, '%s\n');
fName_list = fName_list{1};

fclose(fid);

if(N ~= length(fName_list))
	error(['Number of expected elements incorrect']);
end

end