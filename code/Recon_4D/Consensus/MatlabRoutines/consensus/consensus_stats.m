function [ control_4D, stop_condition ] = consensus_stats(control_4D, itnumber)


if(itnumber~=0)
	switch control_4D.params_consensus.consensus_algo

	case 'CADMM'

		control_4D.stats.runningMetrics.u_fwd_rms(itnumber) =  stat_RMSE_4D( control_4D.binaryFnames_C.u_fwd, []);
	    control_4D.stats.runningMetrics.diff_fwd_before(itnumber) =  stat_RMSE_4D( control_4D.binaryFnames_C.x_fwd, control_4D.binaryFnames_C.x_fwd_before);
	    control_4D.stats.runningMetrics.relUpdatePercent_x_hat(itnumber) = stat_relChange_4D( control_4D.binaryFnames_C.x_fwd, control_4D.binaryFnames_C.x_fwd_before);

	    switch control_4D.params_consensus.consensus_mode
	    case 'time'
	        control_4D.stats.runningMetrics.diff_fwd_cent_prior(itnumber) =  stat_RMSE_4D( control_4D.binaryFnames_C.x_fwd, control_4D.binaryFnames_C.x_cent_prior);
	        control_4D.stats.runningMetrics.rms_fwd(itnumber) =  stat_RMSE_4D( control_4D.binaryFnames_C.x_fwd, []);
	        control_4D.stats.runningMetrics.rms_cent_prior(itnumber) =  stat_RMSE_4D( control_4D.binaryFnames_C.x_cent_prior, []);
	    case 'view'

	    end  

	case 'CE'

		control_4D.stats.runningMetrics.diff_fwd_before(itnumber) =  stat_RMSE_4D( control_4D.binaryFnames_C.x_fwd, control_4D.binaryFnames_C.x_fwd_before);
	    control_4D.stats.runningMetrics.relUpdatePercent_x_hat(itnumber) = stat_relChange_4D( control_4D.binaryFnames_C.x_fwd, control_4D.binaryFnames_C.x_fwd_before);

	    switch control_4D.params_consensus.consensus_mode
	    case 'time'
	        control_4D.stats.runningMetrics.rmse_fwd_cent_prior(itnumber) =  stat_RMSE_4D( control_4D.binaryFnames_C.x_fwd, control_4D.binaryFnames_C.z_cent_prior);
	        control_4D.stats.runningMetrics.rms_fwd(itnumber) =  stat_RMSE_4D( control_4D.binaryFnames_C.x_fwd, []);
	        
	        for n=1:control_4D.num_decentral_denoisers
	            eval(['control_4D.stats.runningMetrics.rmse_cent_prior_dec_Prior_', num2str(n), '(itnumber) =  stat_RMSE_4D( control_4D.binaryFnames_C.x_dec_Prior(n,:), control_4D.binaryFnames_C.z_cent_prior);']);
	            eval(['control_4D.stats.runningMetrics.rms_dec_Prior_', num2str(n), '(itnumber) =  stat_RMSE_4D( control_4D.binaryFnames_C.x_dec_Prior(n,:), []);']);
	        end
	        control_4D.stats.runningMetrics.rms_cent_prior(itnumber) =  stat_RMSE_4D( control_4D.binaryFnames_C.z_cent_prior, []);
	    case 'view'

	    end

	otherwise
	    error('consensus_init_variables: invarid consensus_algo')

	end
end


disp(' === Check stopping condition ===')  
if( itnumber~=0 && control_4D.stats.runningMetrics.relUpdatePercent_x_hat(itnumber) < control_4D.params_consensus.StopThresholdPercent )          
    disp('Reached Stopping Condition!')
    stop_condition = 1;
else
	stop_condition = 0;
end


return
