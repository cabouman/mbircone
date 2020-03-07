function [time, control_4D] = set_sigma( control_4D, plainParamsFile )


% 
%       sigma_lambda <- sigma_ref / sqrt(inversion_priorWeight)
%       sigma_denoisers = sigma_ref * sqrt(priorWeight_denoisers)
% 

tic

if(control_4D.params_consensus.isEstimate_sigma_ref)
    control_4D.params_consensus.sigma_ref = mean(control_4D.std_inv_list(:));
else
    % this uses user-specified value
end
sigma_ref = control_4D.params_consensus.sigma_ref;

control_4D.params_consensus.sigma_lambda = 						sigma_ref * sqrt(control_4D.params_consensus.inversion_priorWeight);
control_4D.params_consensus.decentralized_sigma_denoiser_List = sigma_ref * sqrt(control_4D.params_consensus.decentralized_priorWeightList);

% scale sigma appropriately due to averaging
switch control_4D.params_consensus.consensus_mode
case 'time'
	control_4D.params_consensus.centralized_sigma_denoiser = 		sigma_ref * sqrt(control_4D.params_consensus.centralized_priorWeight / (control_4D.num_decentral_denoisers+1) );
case 'view'
	control_4D.params_consensus.centralized_sigma_denoiser = 		sigma_ref * sqrt(control_4D.params_consensus.centralized_priorWeight / (control_4D.num_decentral_denoisers+control_4D.numVols_inversion) );
end

% Write those params to file
for t=1:control_4D.numVols_inversion
    setReconParamsField('sigma_lambda', num2str(control_4D.params_consensus.sigma_lambda), control_4D.invMasterList{t}, plainParamsFile);
end

if ~isempty(control_4D.DenoisingConfig_central.DenoisingConfigFname)
	addpath(control_4D.DenoisingConfig_central.iopath);
	writeDenoiserSigma(control_4D.params_consensus.centralized_sigma_denoiser, control_4D.DenoisingConfig_central.ScriptArg, plainParamsFile);
	rmpath(control_4D.DenoisingConfig_central.iopath);
end

for n=1:control_4D.num_decentral_denoisers
    addpath(control_4D.DenoisingConfigList_decentral(n).iopath);
    writeDenoiserSigma(control_4D.params_consensus.decentralized_sigma_denoiser_List(n), control_4D.DenoisingConfigList_decentral(n).ScriptArg, plainParamsFile);
    rmpath(control_4D.DenoisingConfigList_decentral(n).iopath);
end


time = toc;
