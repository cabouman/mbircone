function [ ] = writePatchPositionList(fName, lb1, ub1, lb2, ub2, lb3, ub3)

N = length(lb1);

fid = fopen(fName, 'w');

	fprintf(fid, '%d\n', N);
	for i = 1:N
	    fprintf(fid, '%d %d %d %d %d %d\n', lb1(i), ub1(i), lb2(i), ub2(i), lb3(i), ub3(i));
	end

fclose(fid);

end