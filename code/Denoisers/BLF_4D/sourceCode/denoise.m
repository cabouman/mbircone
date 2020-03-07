function [] = denoise(masterFile, plainParamsFile)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../ModularMatlabCode/'));
addpath(fullfile(mfilepath,'../../../Modular_MatlabRoutines/'));
addpath(fullfile(mfilepath,'../../ModularMatlabCode_4D/'));

par = readDenoisingParams(masterFile, plainParamsFile);
[ fileLists ] = readImageFileLists_denoiser_4D( masterFile, plainParamsFile );

printStruct(par, 'par');
fileLists


x = read4D(fileLists.noisyImageNames, 'float32');

disp('Image size:')
disp(size(x))

% to make x in range [0,...,255]
% adjust sigmaR the same way
shift = -min(x(:));
imgrange = range(x(:));
scaler =  1 / imgrange;

x = (x+shift) * scaler;

sigmaS = par.sigmaS ;
sigmaR = par.sigmaR * scaler ;
samS = par.samS ;
samR = 1 / par.numBins ;
verbose = 0 ;

sigmaSx = sigmaS;
sigmaSy = sigmaS;
sigmaSz = sigmaS;
sigmaSt = sigmaS;

samSx = samS;
samSy = samS;
samSz = samS;
samSt = samS;

if size(x,1)==1
	samSx = 1;
	sigmaSx = 1;
end
if size(x,2)==1
	samSy = 1;
	sigmaSy = 1;
end
if size(x,3)==1
	samSz = 1;
	sigmaSz = 1;
end
if size(x,4)==1
	samSt = 1;
	sigmaSt = 1;
end


tic
x_est = bilateral4i(x, sigmaSx, sigmaSy, sigmaSz, sigmaSt, sigmaR ,samSx, samSy, samSz, samSt, samR);
toc

x_est = (x_est / scaler) - shift;
write4D(fileLists.denoisedImageNames, x_est, 'float32');







end