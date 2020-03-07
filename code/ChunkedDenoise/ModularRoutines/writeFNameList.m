function [ ] = writeFNameList(fName, fName_list)

N = length(fName_list);

system(['mkdir -p $(dirname ', fName_list{1}, ')']);

fid = fopen(fName, 'w');

	fprintf(fid, '%d\n', N);
	for i = 1:N
	    fprintf(fid, '%s\n', fName_list{i});
	end

fclose(fid);

end