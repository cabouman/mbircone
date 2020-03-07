function [control_4D, stop_condition] = consensus_step(control_4D, itnumber)

switch control_4D.params_consensus.consensus_algo

case 'CADMM'
    control_4D = CADMM_step( control_4D, itnumber );

case 'CE'
	% stop_condition=1;
	control_4D = CE_step( control_4D, itnumber );

otherwise
    error('consensus_init_variables: invarid consensus_algo')

end

return
