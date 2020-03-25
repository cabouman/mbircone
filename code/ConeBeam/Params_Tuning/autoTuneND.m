function [paramValues_list, error_list] = autoTuneND( masterFile, plainParamsFile, paramNames, paramValues, verbosity)

mfilepath=fileparts(which(mfilename));

addpath(fullfile(mfilepath,'MatlabRoutines'));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));
addpath(fullfile(mfilepath,'../Preprocessing/Modular_PreprocessingRoutines'));
%%
binaryFNames = readBinaryFNames( masterFile, plainParamsFile );
sinoParams = readSinoParams(masterFile, plainParamsFile);
preprocessingParams = readPreprocessingParams(masterFile, plainParamsFile);

%%
opts.verbose = verbosity;
opts.prefix = 'autoTune/'




%%
N_D = length(paramNames); % Number of dimensions
N_values = zeros(N_D, 1);

for i_D = 1:N_D
	N_values(i_D) = length(paramValues{i_D});
end
N_list = prod(N_values);

error_list = zeros(N_list, 1);
paramValues_list = zeros(N_list, N_D);

for index_list = 1:N_list
   
	disp(['>>>>>>>>>>>>>>> Progress autoTuneND = ', num2str((index_list-1)/N_list*100), '%  <<<<<<<<<<<<<<<'])

	% Convert to mulitdimensional index
	field_idxs = arrayIndicesFromLinearIndex(index_list, N_values);

	% Store shift in paramValues_list
	for i_D = 1:N_D
		paramValue = paramValues{i_D}(field_idxs(i_D));
		paramValues_list(index_list, i_D) = paramValue;
	end

	paramNames_single = paramNames;
	paramValues_single = paramValues_list(index_list, :)';

	disp(['paramNames_single =  ', paramNames_single']);
	disp(['paramValues_single'' = ', num2str(paramValues_single')]);


	error_list(index_list) = autoTune_generic( masterFile, plainParamsFile, paramNames_single, paramValues_single, opts);

	disp('error_list'' = ')
	error_list'
	disp('paramValues_list'' = ')
	paramValues_list'

    
end





% 
% 
% % Find minimum
% [~, index_list_min] = min(error_list);
% paramValues_min = paramValues_list(index_list_min, :);
% 
% % Print Result
% for i_D = 1:N_D
% 
%   name = paramNames{i_D};
%   paramValue = paramValues_min(i_D);
%   disp(['Best value for ', name, ' = ', num2str(paramValue)]);
% 
% end
% 
% 
% % store results as string
% str = '';
% for index_list = 1:N_list
%   str = [str, 'paramValues_list(' ,num2str(index_list), ', :) = [', num2str(paramValues_list(index_list,:)), '];']; 
% end
% str = [str, 'error_list = [', num2str(error_list'), '];'];
% str = [str, 'error_list = error_list''; '];
% disp(str);
% 









