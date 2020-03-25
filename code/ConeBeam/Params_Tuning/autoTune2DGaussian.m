%function [] = autoTune2DGaussian( masterFile, plainParamsFile )
clear all
plainParamsFile = '/scratch/snyder/t/tbalke/coneBeam/code/plainParams/plainParams.sh'
masterFile = '/scratch/snyder/t/tbalke/coneBeam/control/Inversion/QGGMRF/master.txt'

mfilepath=fileparts(which(mfilename));



addpath(fullfile(mfilepath,'MatlabRoutines'));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));
addpath(fullfile(mfilepath,'../Preprocessing/Modular_PreprocessingRoutines'));


binaryFNames = readBinaryFNames( masterFile, plainParamsFile );
sinoParams = readSinoParams(masterFile, plainParamsFile);
preprocessingParams = readPreprocessingParams(masterFile, plainParamsFile);





clear paramNames params_init opts 
errList = [];
paramsList = [];
opts.verbose = 0;






N = 5;



paramNames{1} = 'Delta_v_d0';
r = 0.05;
c = 0.50;
params_init{1} = linspace(-r, r, N) + c;

paramNames{2} = 'Delta_w_d0';
r = 20; %20 per degree
c = 0;
params_init{2} = linspace(-r, r, N) + c;

prefix = 'autoTune/';

fileIndex = 1;
for i1 = 1:length(params_init{1})
	for i2 = 1:length(params_init{2})

		disp(sprintf('i1, i2 = %d, %d', i1, i2))
		params = [ params_init{1}(i1), params_init{2}(i2) ]';
		opts.prefix = [prefix, sprintf('%04d_', fileIndex)];
		err = autoTune_generic( masterFile, plainParamsFile, paramNames, params, opts);

		errList = [ errList; err ];
		paramsList = [ paramsList; params' ];

		save('gaussian_errList_paramsList.mat', 'errList', 'paramsList', 'paramNames');

		fileIndex = fileIndex + 1;
	end
end

 
%comment% % Find minimum
%comment% [~, index_linear_min] = min(eList);
%comment% s_min = sList(index_linear_min, :);
%comment% 
%comment% % Print Result
%comment% for i_D = 1:N_D
%comment% 
%comment% 	name = names{i_D};
%comment% 	s = s_min(i_D);
%comment% 	disp(['Best value for ', name, ' = ', num2str(s)]);
%comment% 
%comment% end
%comment% 
%comment% 
%comment% % store results as string
%comment% str = '';
%comment% for index_linear = 1:N_samples
%comment% 	str = [str, 'sList(',num2str(index_linear),', :) = [', num2str(sList(index_linear,:)''), ']; '];
%comment% end
%comment% str = [str, 'eList = [', num2str(eList'), '];'];
%comment% str = [str, 'eList = eList''; '];
%comment% disp(str);
%comment% 
%comment% 
