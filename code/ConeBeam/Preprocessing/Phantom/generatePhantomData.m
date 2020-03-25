function [] = generatePhantomData(masterFile, plainParamsFile)


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../Modular_PreprocessingRoutines'));
addpath(fullfile(mfilepath,'../../../Modular_MatlabRoutines'));

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Read File Names and such
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Read File Names ...');
binaryFNames = readBinaryFNames( masterFile, plainParamsFile );
preprocessingParams = readPreprocessingParams(masterFile, plainParamsFile);
sinoParams = readSinoParams(masterFile, plainParamsFile);
imgParams = readImgParams(masterFile, plainParamsFile);

printStruct(binaryFNames, 'binaryFNames');
printStruct(preprocessingParams, 'preprocessingParams');
printStruct(sinoParams, 'sinoParams');
printStruct(imgParams, 'imgParams');


%% Create Phantom
N = max([imgParams.N_x, imgParams.N_y, imgParams.N_z]);
ph = phantom3d(N);
ph = ph / 50; % <-- This dirty line to get reasonable sinogram values

n = imgParams.N_x;
r_x = floor((N-n)/2)+1:floor((N-n)/2)+n;

n = imgParams.N_y;
r_y = floor((N-n)/2)+1:floor((N-n)/2)+n;

n = imgParams.N_z;
r_z = floor((N-n)/2)+1:floor((N-n)/2)+n;

ph = ph(r_z, r_y, r_x);

write3D(binaryFNames.phantom, ph, 'float32');


%% Create Ideal forward projection
command = ['bash ../../../Inversion/run/genSysMatrix.sh ', masterFile];
[exit_status, commandLineOutput] = system(command);


command = ['bash ../../../Inversion/run/forwardProject.sh -M ', masterFile, ' -i ', binaryFNames.phantom, ' -o ', binaryFNames.sino];
[exit_status, commandLineOutput] = system(command);



sino = read3D(binaryFNames.sino, 'float32');

wght = exp(-sino);

%% Adding noise would be possible here
%  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
%%


%% write to disk
write3D(binaryFNames.sino, sino, 'float32');
write3D(binaryFNames.wght, wght, 'float32');



end

