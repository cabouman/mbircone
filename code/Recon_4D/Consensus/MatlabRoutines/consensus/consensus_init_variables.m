function time = consensus_init_variables( control_4D )

switch control_4D.params_consensus.consensus_algo

case 'CADMM'
	time = CADMM_init_variables( control_4D );

case 'CE'
	time = CE_init_variables( control_4D );

otherwise
	error('consensus_init_variables: invarid consensus_algo')

end

return