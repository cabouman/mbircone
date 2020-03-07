function time = set_proxmapMode( control_4D, plainParamsFile )

% set prior mode and stop criterion in recon params 

tic

for t=1:control_4D.numVols_inversion
	if(control_4D.params_consensus.isFoldedInPrior==0)
	    setReconParamsField('priorWeight_QGGMRF', '-1', control_4D.invMasterList{t}, plainParamsFile);
	else
	    setReconParamsField('priorWeight_QGGMRF', num2str(1/control_4D.params_4D.num_viewSubsets), control_4D.invMasterList{t}, plainParamsFile);
	end
    setReconParamsField('priorWeight_proxMap', '1', control_4D.invMasterList{t}, plainParamsFile);

	%setReconParamsField('stopThresholdChange_pct', '0', control_4D.invMasterList{t}, plainParamsFile);
	%setReconParamsField('stopThesholdRWFE_pct', '0', control_4D.invMasterList{t}, plainParamsFile);
	%setReconParamsField('stopThesholdRUFE_pct', '0', control_4D.invMasterList{t}, plainParamsFile);
	%setReconParamsField('stopThesholdRRMSE_pct', '0', control_4D.invMasterList{t}, plainParamsFile);
	
    setReconParamsField('MaxIterations', num2str( control_4D.params_consensus.fwdProxmap_MaxIter ), control_4D.invMasterList{t}, plainParamsFile);
	setReconParamsField('isEstimateWeightScaler', '0', control_4D.invMasterList{t}, plainParamsFile);
end


time = toc;
