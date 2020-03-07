function [ control_4D ] = consensus_visualization(control_4D, itnumber)


if(itnumber~=0)
	switch control_4D.params_consensus.consensus_algo

	case 'CADMM'

		write_4D_visualization( control_4D.moduleIOFnames.inv_output_list, ['Visualizations_4D/iter_' num2str(itnumber) ], ['recon_' num2str(itnumber) ], control_4D.params_consensus.visualize_coord);
		switch control_4D.params_consensus.consensus_mode
		case 'time'
		    control_4D.visualizationFnames.x_cent_prior{itnumber} =  write_4D_visualization( control_4D.binaryFnames_C.x_cent_prior, ['Visualizations_4D/iter_' num2str(itnumber) ], ['x_cent_prior_iter_' num2str(itnumber) ], control_4D.params_consensus.visualize_coord);
		    control_4D.visualizationFnames.x_avg{itnumber} = write_4D_visualization( control_4D.binaryFnames_C.x_avg, ['Visualizations_4D/iter_' num2str(itnumber) ], ['x_avg_iter_' num2str(itnumber) ], control_4D.params_consensus.visualize_coord);
		    control_4D.visualizationFnames.u_avg{itnumber} = write_4D_visualization( control_4D.binaryFnames_C.u_avg, ['Visualizations_4D/iter_' num2str(itnumber) ], ['u_avg_iter_' num2str(itnumber) ], control_4D.params_consensus.visualize_coord);

		    control_4D.visualizationFnames.x_fwd{itnumber} = write_4D_visualization( control_4D.binaryFnames_C.x_fwd, ['Visualizations_4D/iter_' num2str(itnumber) ], ['x_fwd_iter_' num2str(itnumber) ], control_4D.params_consensus.visualize_coord);
		    control_4D.visualizationFnames.u_fwd{itnumber} = write_4D_visualization( control_4D.binaryFnames_C.u_fwd, ['Visualizations_4D/iter_' num2str(itnumber) ], ['u_fwd_iter_' num2str(itnumber) ], control_4D.params_consensus.visualize_coord);
		    for n=1:control_4D.num_decentral_denoisers
		        control_4D.visualizationFnames.x_dec_Prior{n}{itnumber} = write_4D_visualization( control_4D.binaryFnames_C.x_dec_Prior(n,:), ['Visualizations_4D/iter_' num2str(itnumber) ], ['x_dec_Prior_n_', num2str(n), '_iter_', num2str(itnumber) ], control_4D.params_consensus.visualize_coord);
		        control_4D.visualizationFnames.u_dec_Prior{n}{itnumber} = write_4D_visualization( control_4D.binaryFnames_C.u_dec_Prior(n,:), ['Visualizations_4D/iter_' num2str(itnumber) ], ['u_dec_Prior_n_', num2str(n), '_iter_', num2str(itnumber) ], control_4D.params_consensus.visualize_coord);
		    end

		case 'view'

		end

	case 'CE'

		write_4D_visualization( control_4D.moduleIOFnames.inv_output_list, ['Visualizations_4D/iter_' num2str(itnumber) ], ['recon_' num2str(itnumber) ], control_4D.params_consensus.visualize_coord);
		switch control_4D.params_consensus.consensus_mode
		case 'time'
		    control_4D.visualizationFnames.z_cent_prior{itnumber} = write_4D_visualization( control_4D.binaryFnames_C.z_cent_prior, ['Visualizations_4D/iter_' num2str(itnumber) ], ['z_cent_prior_iter_' num2str(itnumber) ], control_4D.params_consensus.visualize_coord);

		    control_4D.visualizationFnames.v_avg{itnumber} = write_4D_visualization( control_4D.binaryFnames_C.v_avg, ['Visualizations_4D/iter_' num2str(itnumber) ], ['v_avg_iter_' num2str(itnumber) ], control_4D.params_consensus.visualize_coord);
		    control_4D.visualizationFnames.v_fwd{itnumber} = write_4D_visualization( control_4D.binaryFnames_C.v_fwd, ['Visualizations_4D/iter_' num2str(itnumber) ], ['v_fwd_iter_' num2str(itnumber) ], control_4D.params_consensus.visualize_coord);

		    control_4D.visualizationFnames.w_fwd{itnumber} = write_4D_visualization( control_4D.binaryFnames_C.w_fwd, ['Visualizations_4D/iter_' num2str(itnumber) ], ['w_fwd_iter_' num2str(itnumber) ], control_4D.params_consensus.visualize_coord);
		    control_4D.visualizationFnames.x_fwd{itnumber} = write_4D_visualization( control_4D.binaryFnames_C.x_fwd, ['Visualizations_4D/iter_' num2str(itnumber) ], ['x_fwd_iter_' num2str(itnumber) ], control_4D.params_consensus.visualize_coord);
		    control_4D.visualizationFnames.x_fwd_before{itnumber} = write_4D_visualization( control_4D.binaryFnames_C.x_fwd_before, ['Visualizations_4D/iter_' num2str(itnumber) ], ['x_fwd_before_iter_' num2str(itnumber) ], control_4D.params_consensus.visualize_coord);

		    for n=1:control_4D.num_decentral_denoisers
		        control_4D.visualizationFnames.x_dec_Prior{n}{itnumber} = write_4D_visualization( control_4D.binaryFnames_C.x_dec_Prior(n,:), ['Visualizations_4D/iter_' num2str(itnumber) ], ['x_dec_Prior_n_', num2str(n), '_iter_', num2str(itnumber) ], control_4D.params_consensus.visualize_coord);
		        control_4D.visualizationFnames.v_dec_Prior{n}{itnumber} = write_4D_visualization( control_4D.binaryFnames_C.v_dec_Prior(n,:), ['Visualizations_4D/iter_' num2str(itnumber) ], ['v_dec_Prior_n_', num2str(n), '_iter_', num2str(itnumber) ], control_4D.params_consensus.visualize_coord);
		        control_4D.visualizationFnames.w_dec_Prior{n}{itnumber} = write_4D_visualization( control_4D.binaryFnames_C.w_dec_Prior(n,:), ['Visualizations_4D/iter_' num2str(itnumber) ], ['w_dec_Prior_n_', num2str(n), '_iter_', num2str(itnumber) ], control_4D.params_consensus.visualize_coord);
		    end

		case 'view'

		end

	otherwise
	    error('consensus_init_variables: invarid consensus_algo')

	end
end


return
