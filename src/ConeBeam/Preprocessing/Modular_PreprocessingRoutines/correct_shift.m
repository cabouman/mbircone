function [sino, wght, driftReference_sino, jigMeasurementsSino] = correct_shift( sino, wght, driftReference_sino, jigMeasurementsSino, TotalAngle, viewAngleList, preprocessingParams)

N_beta = size(sino,3);

if preprocessingParams.shift_correctionType == 1

	tic
	[shift1_total, shift2_total] = find_shift_multiscale( driftReference_sino(:,:,1), driftReference_sino(:,:,2), ...
		preprocessingParams.shift_searchRadius, preprocessingParams.shift_numPointsGrid, preprocessingParams.shift_gridReductionRatio, preprocessingParams.shift_gridSize );
	time_find_shift = toc;
	fprintf('Final Shifts: %d %d\n',shift1_total,shift2_total );
	fprintf('Time taken: %d \n',time_find_shift );
end

if preprocessingParams.shift_correctionType == 2
	shift_vect = eval(preprocessingParams.shift_values);
	shift1_total = shift_vect(1);
	shift2_total = shift_vect(2);
	fprintf('Final Shifts: %d %d\n',shift1_total,shift2_total );
end

tic
if preprocessingParams.shift_correctionType == 2 || preprocessingParams.shift_correctionType == 1 
	% Correct for x-ray source shift
	TotalAngle_rad = (2*pi/360) * TotalAngle ;
	mean_angle = ( viewAngleList(1) + viewAngleList(N_beta) )/2 ;
	for i = 1:N_beta
		fraction = ( viewAngleList(i) - mean_angle ) / TotalAngle_rad;
	    shift1_in_view = shift1_total * fraction ;
	    shift2_in_view = shift2_total * fraction ;
	    sino(:,:,i) = shift_img( sino(:,:,i), -shift1_in_view, -shift2_in_view);
	    wght(:,:,i) = shift_img( wght(:,:,i), -shift1_in_view, -shift2_in_view);
	end
	for i = 1:size(jigMeasurementsSino,3)
	    jigMeasurementsSino(:,:,i) = shift_img( jigMeasurementsSino(:,:,i), -shift1_in_view, -shift2_in_view);
	end
	driftReference_sino(:,:,2) = shift_img( sino(:,:,i), -shift1_total, -shift2_total);
end
time_shift = toc;
fprintf('Shifting time: %d\n',time_shift );


% tic
% % x = signal_shift(sino(1,:,i), 0.5, 0, 'fullsinc');
% x = signal_interp(sino(1,:,1), 0.5, 0, 'fullsinc');
% toc 


return