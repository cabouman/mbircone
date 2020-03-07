function [] = chunk(masterFile, plainParamsFile)
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../Modular_MatlabRoutines/'));
addpath(fullfile(mfilepath,'./ModularRoutines/'));

par = readParams(masterFile, plainParamsFile);
printStruct(par, 'par');

x = read3D( par.inputImageFName, par.dataType );

[ x_chunked, lb1, ub1, lb2, ub2, lb3, ub3 ] = split3DArray(x, par.haloRadius, par.maxChunkSize);

%% %%%%%%%%%%%%%%%%% Patch positions

writePatchPositionList(par.patchPositionList, lb1, ub1, lb2, ub2, lb3, ub3);

%% %%%%%%%%%%%%%%%%% Binary files 
N = length(lb1);

[ pid_lb, pid_ub ] = generateProcessIndexList(par.maxNumProcesses, N);

numProcesses = length(pid_lb);

disp(['Number of chunks = ', num2str(N)])
disp(['Number of processes = ', num2str(numProcesses)])
temp = x_chunked{1};
disp(['Chunk Size approximately: ', num2str(size(temp,1)), 'x', num2str(size(temp,2)), 'x', num2str(size(temp,3))])
%% file name list-lists
inputImageListList = generateFNamesList(par.folderSuffix, par.inputImageListList, 1, numProcesses);
outputImageListList = generateFNamesList(par.folderSuffix, par.outputImageListList, 1, numProcesses);

writeFNameList(par.inputImageListList, inputImageListList);
writeFNameList(par.outputImageListList, outputImageListList);

% file name lists

for pid = 1:numProcesses
	i_start = pid_lb(pid);
	i_stop = pid_ub(pid);

	inputImageList = generateFNamesList(par.folderSuffix, par.inputImageFName, i_start, i_stop);
	outputImageList = generateFNamesList(par.folderSuffix, par.outputImageFName, i_start, i_stop);

	writeFNameList(inputImageListList{pid}, inputImageList);
	writeFNameList(outputImageListList{pid}, outputImageList);

	% Write binary files
	for i = i_start:i_stop
		write3D( inputImageList{i-i_start+1}, x_chunked{i} , par.dataType);
	end
end









end