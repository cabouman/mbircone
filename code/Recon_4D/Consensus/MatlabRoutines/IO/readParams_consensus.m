function [ params_consensus ] = readParams_consensus( masterFile, plainParamsFile )

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'params_consensus';
value = '';
resolveFlag = '';

params_consensus.consensus_mode  = plainParams(executablePath, get_set, masterFile, masterField, 'consensus_mode', value, resolveFlag);
params_consensus.consensus_algo  = plainParams(executablePath, get_set, masterFile, masterField, 'consensus_algo', value, resolveFlag);

params_consensus.MaxIter  = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'MaxIter', value, resolveFlag));
params_consensus.StopThresholdPercent  = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'StopThresholdPercent', value, resolveFlag));
params_consensus.fwdProxmap_MaxIter  = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'fwdProxmap_MaxIter', value, resolveFlag));
params_consensus.isInitializeRecon  = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'isInitializeRecon', value, resolveFlag));
params_consensus.isInitializeVars  = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'isInitializeVars', value, resolveFlag));
params_consensus.isConsensusIters  = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'isConsensusIters', value, resolveFlag));
params_consensus.forceStopFlag  = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'forceStopFlag', value, resolveFlag));

params_consensus.clearBinaryFolder_start  = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'clearBinaryFolder_start', value, resolveFlag));
params_consensus.clearBinaryFiles_end  = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'clearBinaryFiles_end', value, resolveFlag));

params_consensus.isParallel_inv_denoising_dec  = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'isParallel_inv_denoising_dec', value, resolveFlag));

params_consensus.testDenoiser  = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'testDenoiser', value, resolveFlag));

params_consensus.rho  = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'rho', value, resolveFlag));

params_consensus.isEstimate_sigma_ref  = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'isEstimate_sigma_ref', value, resolveFlag));
params_consensus.sigma_ref  = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'sigma_ref', value, resolveFlag));
params_consensus.inversion_priorWeight  = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'inversion_priorWeight', value, resolveFlag));
params_consensus.centralized_priorWeight  = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'centralized_priorWeight', value, resolveFlag));

decentralized_priorWeightList_str  = plainParams(executablePath, get_set, masterFile, masterField, 'decentralized_priorWeightList', value, resolveFlag);
params_consensus.decentralized_priorWeightList  = eval(decentralized_priorWeightList_str);

averaging_wtList_relative_str  = plainParams(executablePath, get_set, masterFile, masterField, 'averaging_wtList_relative', value, resolveFlag);
averaging_wtList_relative = eval(averaging_wtList_relative_str);
params_consensus.averaging_wtList = averaging_wtList_relative/sum(averaging_wtList_relative(:));

params_consensus.isFoldedInPrior = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'isFoldedInPrior', value, resolveFlag));

params_consensus.visualize_coord = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'visualize_coord', value, resolveFlag));

return

