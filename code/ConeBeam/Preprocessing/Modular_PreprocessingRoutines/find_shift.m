function [shift1_new, shift2_new] = find_shift( scan1, scan2, shift1, shift2, radius, stepSize )

% Finds shift between two scans
% can handle two scans with multiple views in them

s1_list = shift1-radius:stepSize:shift1+radius ;
s2_list = shift2-radius:stepSize:shift2+radius ;

N1 = length(s1_list);
N2 = length(s2_list);


error_list = zeros(N1,N2);

for i=1:N1
	for j=1:N2
		s1 = s1_list(i);
		s2 = s2_list(j);

		% shift scans
		shifted_scan1 = shift_img(scan1, s1, s2);
		error_list(i,j) = immse(shifted_scan1, scan2);
	end
end
% disp('s1_list')
% disp(s1_list)
% disp('s2_list')
% disp(s2_list)

% disp('error_list')
% disp(error_list)


[min_val, min_id] = min(error_list(:));
% fprintf('Min val: %d\n',min_val );
% fprintf('Min id: %d\n',min_id );
[min_id1, min_id2] = ind2sub([N1 N2], min_id);
% fprintf('Min ids: %d %d\n',min_id1, min_id2 );
shift1_new = s1_list(min_id1);
shift2_new = s2_list(min_id2);

fprintf('Found Shifts: %d %d. Error value: %d \n',shift1_new, shift2_new, min_val );
 
return