function [par] = adjustParameters_axisTilt(par, preprocessingParams)


angle_degree = preprocessingParams.Delta_axis_tilt;

if angle_degree~=0

	rotation_center_v = par.v_r;
	rotation_center_w = 0;

	detector_center_v = par.v_d0 + par.N_dv * par.Delta_dv / 2;
	detector_center_w = par.w_d0 + par.N_dw * par.Delta_dw / 2;

	origin_shift_vect = [ rotation_center_v; rotation_center_w ] - [ detector_center_v; detector_center_w ];

	rot_angle = -preprocessingParams.Delta_axis_tilt;
	rotation_matrix = [ cosd(rot_angle), -sind(rot_angle); sind(rot_angle), cosd(rot_angle) ];

	net_shift_vect = [eye(2) - rotation_matrix]*origin_shift_vect;

	disp('Net shift v:');
	disp(net_shift_vect(1));
	disp('Net shift w:');
	disp(net_shift_vect(2));


	par.v_d0 = par.v_d0 + net_shift_vect(1);
	par.w_d0 = par.w_d0 + net_shift_vect(2);

end


end