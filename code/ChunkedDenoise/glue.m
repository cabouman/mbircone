function [] = glue(masterFile, plainParamsFile)
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../Modular_MatlabRoutines/'));
addpath(fullfile(mfilepath,'./ModularRoutines/'));

par = readParams(masterFile, plainParamsFile);
printStruct(par, 'par');

[lb1, ub1, lb2, ub2, lb3, ub3] = readPatchPositionList(par.patchPositionList);


outputImageListList = readFNameList(par.outputImageListList);
numProcesses = length(outputImageListList);

i_start = 0;
i_stop = 0;
for pid = 1:numProcesses

	outputImageList = readFNameList(outputImageListList{pid});

	i_start = i_stop + 1;
	i_stop = i_start + length(outputImageList) - 1;

	for i = i_start:i_stop
		x_chunked{i} = read3D( outputImageList{i-i_start+1} , par.dataType);
	end
end

N = length(x_chunked);

disp(['Number of cunks = ', num2str(N)])
disp(['Number of processes = ', num2str(numProcesses)])


N1_total = max(ub1);
N2_total = max(ub2);
N3_total = max(ub3);

x_glued = zeros([N1_total, N2_total, N3_total]);
w_glued = zeros(size(x_glued));

for i = 1:N
    
	N1 = ub1(i) - lb1(i) + 1;
	N2 = ub2(i) - lb2(i) + 1;
	N3 = ub3(i) - lb3(i) + 1;

    w_piece = triangleND([N1, N2, N3]);
    x_piece = x_chunked{i} .* w_piece;
    
    x_glued(lb1(i):ub1(i), lb2(i):ub2(i), lb3(i):ub3(i)) =  x_glued(lb1(i):ub1(i), lb2(i):ub2(i), lb3(i):ub3(i)) + x_piece;
    w_glued(lb1(i):ub1(i), lb2(i):ub2(i), lb3(i):ub3(i)) =  w_glued(lb1(i):ub1(i), lb2(i):ub2(i), lb3(i):ub3(i)) + w_piece;
    
    
end
    
x_recon = x_glued ./ w_glued;

write3D( par.outputImageFName, x_recon , par.dataType);


end