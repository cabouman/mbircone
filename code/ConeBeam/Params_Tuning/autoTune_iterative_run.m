% clear all

mfilepath=fileparts(which(mfilename));
% disp(['mfilepath: ' mfilepath]);

addpath(fullfile(mfilepath,'MatlabRoutines'));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));
addpath(fullfile(mfilepath,'../Preprocessing/Modular_PreprocessingRoutines'));
addpath(fullfile(mfilepath,'../../plainParams/'));


mu = str2num(plainParams(plainParamsFile, 'get', masterFile, 'preprocessingParams', paramName, '', ''))


% -------------------------------
for it = 1:length(searchRadius_list)

    disp(['---------------------------------- Iteration ', num2str(it), ' of ', num2str(length(searchRadius_list)), ' ----------------------------------']);

    plainParams(plainParamsFile, 'set', masterFile, 'preprocessingParams', paramName, num2str(mu), '');
    searchRadius = searchRadius_list(it);
    numPointsGrid = numPointsGrid_list(it);
    midPoint = str2num(plainParams(plainParamsFile, 'get', masterFile, 'preprocessingParams', paramName, '', ''));

    paramNames{1,1} = paramName;
    paramValues{1,1} = linspace(midPoint-searchRadius, midPoint+searchRadius, numPointsGrid)';


    tic
    [paramValues_list, error_list] = autoTuneND( masterFile, plainParamsFile,  paramNames, paramValues, verbosity);
    toc
    str = '';
    str = [str, 'paramValues_list = [', num2str(paramValues_list'),']; paramValues_list = paramValues_list''; '];
    str = [str, 'error_list = [', num2str(error_list'),']; error_list = error_list''; '];
    disp(str);

    f = error_list;
    X = paramValues_list;
    W = ones(size(error_list));


    [ b, mu, c] = approximate1DQuadraticLS( f, X, W);
    mu

    % Find minimum
    [~, index_list_min] = min(error_list);
    paramValues_min = paramValues_list(index_list_min, :);
    paramValues_min

end

plainParams(plainParamsFile, 'set', masterFile, 'preprocessingParams', paramName, num2str(mu), '');


% -------------------------------





