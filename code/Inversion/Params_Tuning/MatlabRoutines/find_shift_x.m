function [shift_new, error_list, s_list, error_scan] = find_shift_x( scan1, scan2, wght, shift, radius, stepSize )

% Finds shift between two scans
% can handle two scans with multiple views in them

s_list = (shift-radius:stepSize:shift+radius)' ;

N = length(s_list);


error_list = zeros(N,1);

for i=1:N
	s = s_list(i);

	% shift scans
	shifted_scan1 = shift_img(scan1, 0, s);
	error_scan = shifted_scan1 - scan2;
	% error_list(i) = immse( shifted_scan1, scan2 );
	error_scan_energy = error_scan.^2 .* wght;
	error_list(i) = sum(error_scan_energy(:));
end

% disp('s_list')
% disp(s_list)
% disp('error_list')
% disp(error_list)


[min_val, min_id] = min(error_list(:));
shift_new = s_list(min_id);

% get best error scan
shifted_scan1 = shift_img(scan1, 0, shift_new);
error_scan = shifted_scan1 - scan2;


fprintf('Found Shift_x: %d. Error value: %d \n',shift_new, min_val );
 
return