function [ ] = multiResolution(masterFile, plainParamsFile)



mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));
addpath(fullfile(mfilepath,'../../plainParams/'));

opts.verbosity = 1;
%% ------------- params

bin_min = str2num(plainParams(plainParamsFile, 'get', masterFile, 'preprocessingParams', 'downscale_dv', '', ''));
binning_arr = [4 2 1] * bin_min

N_beta_max = str2num(plainParams(plainParamsFile, 'get', masterFile, 'preprocessingParams', 'N_beta_all', '', ''));


N_beta_arr = round(N_beta_max ./ sqrt(binning_arr / binning_arr(end)))

%% -------------


numRecons = length(binning_arr);

for i = 1:numRecons
	tic

	disp(['----- MulitResolution iteration ', num2str(i), ' of ', num2str(numRecons), ' -----']);
	disp('Running with:')
	disp(['--- Binning = ', num2str(binning_arr(i))])
	disp(['--- N_beta = ', num2str(N_beta_arr(i))])

	plainParams(plainParamsFile, 'set', masterFile, 'preprocessingParams', 'N_beta_all', num2str(N_beta_arr(i)), '');
	plainParams(plainParamsFile, 'set', masterFile, 'preprocessingParams', 'downscale_dv', num2str(binning_arr(i)), '');
	plainParams(plainParamsFile, 'set', masterFile, 'preprocessingParams', 'downscale_dw', num2str(binning_arr(i)), '');

	if(i == 1)
		if(opts.verbosity>0)
			[exitStatus, ~] = system(['../run/preprocessing.sh ', masterFile], '-echo');
		else
			[exitStatus, ~] = system(['../run/preprocessing.sh ', masterFile]);
		end
		if(exitStatus~=0) error('system command failed'); end

		if(opts.verbosity>0)
			[exitStatus, ~] = system(['../run/runall.sh ', masterFile], '-echo');
		else
			[exitStatus, ~] = system(['../run/runall.sh ', masterFile]);
		end
		if(exitStatus~=0) error('system command failed'); end

	else
		if(opts.verbosity>0)
			[exitStatus, ~] = system(['../run/changePreprocessing.sh ', masterFile], '-echo');
		else
			[exitStatus, ~] = system(['../run/changePreprocessing.sh ', masterFile]);
		end
		if(exitStatus~=0) error('system command failed'); end


		if(opts.verbosity>0)
			[exitStatus, ~] = system(['../run/Recon.sh ', masterFile], '-echo');
		else
			[exitStatus, ~] = system(['../run/Recon.sh ', masterFile]);
		end
		if(exitStatus~=0) error('system command failed'); end
	end

	binaryFNames = readBinaryFNames( masterFile, plainParamsFile );

	[ fName_recon_new ] = prependAppendFileName( 'multiResolution/', binaryFNames.recon, ['.bin=', num2str(binning_arr(i)), '_N_beta=', num2str(N_beta_arr(i))]);
	[ fName_errsino_new ] = prependAppendFileName( 'multiResolution/', binaryFNames.errsino, ['.bin=', num2str(binning_arr(i)), '_N_beta=', num2str(N_beta_arr(i))]);
	

	copyfile(binaryFNames.recon, fName_recon_new);
	copyfile(binaryFNames.errsino, fName_errsino_new);

	elapsed = toc;
	disp(['Time for multiResolution iteration ', num2str(i), ' = ', num2str(elapsed), '.'])

end











end
