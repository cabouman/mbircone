function control_4D = read_control_4D(masterFile, plainParamsFile)

control_4D.master_4D = masterFile;
control_4D.params_4D = readParams_4D( masterFile, plainParamsFile );
control_4D.invMasterList = readInvMasterList( masterFile, plainParamsFile );
control_4D.inversionConfig = readInversionConfig( masterFile, plainParamsFile );
control_4D.DenoisingConfig_central = readDenoisingConfig_central( masterFile, plainParamsFile );
control_4D.DenoisingConfigList_decentral = readDenoisingConfigList_decentral( masterFile, plainParamsFile );
control_4D.statsFile = readStatsFile( masterFile, plainParamsFile );

control_4D.params_consensus = readParams_consensus( masterFile, plainParamsFile );

switch control_4D.params_consensus.consensus_mode
case 'time'

    control_4D.numVols_inversion = length(control_4D.invMasterList);
    control_4D.numVols_denoising = length(control_4D.invMasterList);
    control_4D.num_decentral_denoisers = length(control_4D.DenoisingConfigList_decentral);

case 'view'

    control_4D.numVols_inversion = length(control_4D.invMasterList);
    control_4D.numVols_denoising = 1;
    control_4D.num_decentral_denoisers = length(control_4D.DenoisingConfigList_decentral);

end    

switch control_4D.params_consensus.consensus_algo
case 'CADMM'
	control_4D.binaryFroots_C = readBinaryFroots_CADMM( masterFile, plainParamsFile );
	control_4D.binaryFnames_C = CADMM_genBinaryFileNames( control_4D );

case 'CE'
	control_4D.binaryFroots_C = readBinaryFroots_CE( masterFile, plainParamsFile );
	control_4D.binaryFnames_C = CE_genBinaryFileNames( control_4D );

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% module io file names for inv, denoiser_dec, denoiser_cent
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

control_4D.moduleIOFnames.inv_input_list = get_Inversion_FNameList( control_4D.invMasterList, plainParamsFile, 'proxMapInput' );
control_4D.moduleIOFnames.inv_output_list = get_Inversion_FNameList( control_4D.invMasterList, plainParamsFile, 'recon' );

if ~isempty(control_4D.DenoisingConfig_central.DenoisingConfigFname)
    addpath(control_4D.DenoisingConfig_central.iopath);
    [ noisyFname_list, denoisedFname_list ] =  generateBinaryFNames_denoiser_4D( control_4D.DenoisingConfig_central.ScriptArg, ...
            plainParamsFile, control_4D.numVols_denoising, control_4D.params_4D.prefixRoot, control_4D.params_4D.suffixRoot );
    control_4D.moduleIOFnames.denoiser_cent_noisy_list = noisyFname_list;
    control_4D.moduleIOFnames.denoiser_cent_denoised_list = denoisedFname_list;
end

control_4D.moduleIOFnames.denoiser_dec_noisy_list = cell(control_4D.num_decentral_denoisers, control_4D.numVols_denoising);
control_4D.moduleIOFnames.denoiser_dec_denoised_list = cell(control_4D.num_decentral_denoisers, control_4D.numVols_denoising);
for n=1:control_4D.num_decentral_denoisers
    addpath(control_4D.DenoisingConfigList_decentral(n).iopath);
    [ noisyFname_list, denoisedFname_list ] =  generateBinaryFNames_denoiser_4D( control_4D.DenoisingConfigList_decentral(n).ScriptArg, ...
        plainParamsFile, control_4D.numVols_denoising, control_4D.params_4D.prefixRoot, control_4D.params_4D.suffixRoot );
    control_4D.moduleIOFnames.denoiser_dec_noisy_list(n,:) = noisyFname_list;
    control_4D.moduleIOFnames.denoiser_dec_denoised_list(n,:) = denoisedFname_list;
end

return

