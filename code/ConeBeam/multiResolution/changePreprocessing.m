function [  ] = changePreprocessing( masterFile, plainParamsFile)

mfilepath=fileparts(which(mfilename));

addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));
%%
binaryFNames = readBinaryFNames( masterFile, plainParamsFile );

copyfile(binaryFNames.recon, binaryFNames.phantom);


[exitStatus, ~] = system(['../run/preprocessing.sh ', masterFile], '-echo');
if(exitStatus~=0) error('system command failed'); end


imgParams = readImgParams(masterFile, plainParamsFile);

x_old = read3D(binaryFNames.phantom, 'float32');
x_new = imresize3(x_old,[imgParams.N_z imgParams.N_y imgParams.N_x]);

write3D(binaryFNames.phantom, x_new, 'float32');



isPhantomPresent_backup = plainParams(plainParamsFile, 'get', masterFile, 'reconParams', 'isPhantomPresent', '', '');
isUsePhantomToInitErrSino_backup = plainParams(plainParamsFile, 'get', masterFile, 'reconParams', 'isUsePhantomToInitErrSino', '', '');


plainParams(plainParamsFile, 'set', masterFile, 'reconParams', 'isPhantomPresent', '1', '');
plainParams(plainParamsFile, 'set', masterFile, 'reconParams', 'isUsePhantomToInitErrSino', '1', '');



[exitStatus, ~] = system(['../run/runall_exceptRecon.sh ', masterFile], '-echo');

plainParams(plainParamsFile, 'set', masterFile, 'reconParams', 'isPhantomPresent', isPhantomPresent_backup, '');
plainParams(plainParamsFile, 'set', masterFile, 'reconParams', 'isUsePhantomToInitErrSino', isUsePhantomToInitErrSino_backup, '');

if(exitStatus~=0) error('system command failed'); end

end
