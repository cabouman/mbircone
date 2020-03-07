function [] = print_control_4D( control_4D )

fprintf('numVols_inversion: %d\n',control_4D.numVols_inversion );
fprintf('numVols_denoising: %d\n',control_4D.numVols_denoising );
fprintf('num_decentral_denoisers: %d\n',control_4D.num_decentral_denoisers );

disp('invMasterList:')
for t=1:control_4D.numVols_inversion
	disp(control_4D.invMasterList{t})
end

printStruct(control_4D.params_4D, 'params_4D');
printStruct(control_4D.inversionConfig, 'inversionConfig');
printStruct(control_4D.DenoisingConfig_central, 'DenoisingConfig_central');
if length(control_4D.DenoisingConfigList_decentral) ~= 0
    printStruct(control_4D.DenoisingConfigList_decentral, 'DenoisingConfigList_decentral');
end
printStruct(control_4D.params_consensus, 'params_consensus');
printStruct(control_4D.binaryFroots_C, 'binaryFroots_C');


return