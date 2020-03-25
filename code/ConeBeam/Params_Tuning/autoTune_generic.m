function [errorMetric] = autoTune_generic( masterFile, plainParamsFile, paramNames, paramValues, opts)

mfilepath=fileparts(which(mfilename));

addpath(fullfile(mfilepath,'MatlabRoutines'));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));
addpath(fullfile(mfilepath,'../Preprocessing/Modular_PreprocessingRoutines'));

binaryFNames = readBinaryFNames( masterFile, plainParamsFile );
sinoParams = readSinoParams(masterFile, plainParamsFile);
preprocessingParams = readPreprocessingParams(masterFile, plainParamsFile);

preprecessingCommand = ['bash ../run/preprocessing.sh ', masterFile];
reconCommand = ['bash ../run/runall.sh ', masterFile];




% Number of dimensions
N_D = length(paramNames);

% Set preprocessing paramValues up
for i_D = 1:N_D

	plainParams(plainParamsFile, 'set', masterFile, 'preprocessingParams', paramNames{i_D}, num2str(paramValues(i_D)), '');

end


% Preprocessing and Recon
if(opts.verbose == 1)
	[exit_status, commandLineOutput] = system(preprecessingCommand, '-echo');
	if(exit_status ~= 0)
		disp(commandLineOutput);
		error(['Error executing command ', '"', command, '"']);
	end


	[exit_status, commandLineOutput] = system(reconCommand, '-echo');
	if(exit_status ~= 0)
		disp(commandLineOutput);
		error(['Error executing command ', '"', command, '"']);
	end

else
	[exit_status, commandLineOutput] = system(preprecessingCommand);
	if(exit_status ~= 0)
		disp(commandLineOutput);
		error(['Error executing command ', '"', command, '"']);
	end

	[exit_status, commandLineOutput] = system(reconCommand);	
	if(exit_status ~= 0)
		disp(commandLineOutput);
		error(['Error executing command ', '"', command, '"']);
	end
	
end

% Compute and store error
errsino = read3D( binaryFNames.errsino, 'float32');
wght = read3D( binaryFNames.wght, 'float32');
errorMetric = sum(errsino(:).*wght(:).*errsino(:))/length(errsino(:));


% Create file name suffix
suffix = '';
for i_D = 1:N_D
	suffix = [suffix, sprintf('_%s=%s', paramNames{i_D}, num2str(paramValues(i_D)))];
end


% Store downsampled version of errsino and recon
recon = read3D( binaryFNames.recon, 'float32');
write3D(prependAppendFileName( opts.prefix, binaryFNames.recon, suffix), recon(1:10:end,:,:), 'float32');

errsino = read3D( binaryFNames.errsino, 'float32');
write3D(prependAppendFileName( opts.prefix, binaryFNames.errsino, suffix), errsino(:,:,1:10:end), 'float32');


