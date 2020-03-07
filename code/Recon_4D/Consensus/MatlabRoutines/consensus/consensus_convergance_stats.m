function [ control_4D ] = consensus_convergance_stats(control_4D)



switch control_4D.params_consensus.consensus_algo

case 'CADMM'


case 'CE'

    switch control_4D.params_consensus.consensus_mode
    case 'time'
    	numIters = length(control_4D.visualizationFnames.z_cent_prior);
    	for itnumber=1:numIters
    		currentVolsList{1} = control_4D.visualizationFnames.z_cent_prior{itnumber};
    		convergedVolsList{1} = control_4D.visualizationFnames.z_cent_prior{numIters};
        	control_4D.stats.convergedMetrics.diff_z_cent_prior(itnumber) = stat_RMSE_4D( currentVolsList, convergedVolsList);
        end

    case 'view'

    end

otherwise
    error('consensus_init_variables: invarid consensus_algo')

end


return
