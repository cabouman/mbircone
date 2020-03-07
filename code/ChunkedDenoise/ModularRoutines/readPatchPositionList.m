function [lb1, ub1, lb2, ub2, lb3, ub3] = readPatchPositionList(fName)


fid = fopen(fName, 'r');

	N = fscanf(fid, '%d\n', [1,1]);
	temp = fscanf(fid, '%d %d %d %d %d %d', [6,N]);

fclose(fid);


lb1 = temp(1,:);
ub1 = temp(2,:);
lb2 = temp(3,:);
ub2 = temp(4,:);
lb3 = temp(5,:);
ub3 = temp(6,:);
end
