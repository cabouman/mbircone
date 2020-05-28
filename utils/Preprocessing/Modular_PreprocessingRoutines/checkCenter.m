function [] = checkCenter(par)

cent_v = 0 - (par.v_d0 / (par.Delta_dv * par.N_dv));
cent_w = 0 - (par.w_d0 / (par.Delta_dw * par.N_dw));

disp(['Projection center at (v,w) <-> (', num2str(cent_v), ', ', num2str(cent_w), ').']);
disp('   (Where (v,w) <-> (0, 0) to (1, 1) corresponds to the detector boundaries)');

if(abs(cent_v-0.5) > 0.5)
	error('Projection of rotation axis outside the detector');
end

if(abs(cent_w-0.5) >0.5)
	warning('Projection center does not fall within detector');
end



end