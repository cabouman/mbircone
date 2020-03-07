function [ ] = crossCheck_control_4D(control_4D)

if length(control_4D.params_consensus.decentralized_priorWeightList) ~= length(control_4D.DenoisingConfigList_decentral)
    error('crossCheck_control_4D: decentralized_priorWeightList DenoisingConfigList_decentral lengths dont match');
end

if length(control_4D.params_consensus.averaging_wtList)-1 ~= length(control_4D.DenoisingConfigList_decentral)
    error('crossCheck_control_4D: averaging_wtList DenoisingConfigList_decentral lengths dont differ by 1');
end

return