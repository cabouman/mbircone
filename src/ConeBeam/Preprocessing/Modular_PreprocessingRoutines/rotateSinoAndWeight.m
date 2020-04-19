function [sino, wght, jigMeasurementsSino] = rotateSinoAndWeight( sino, wght, jigMeasurementsSino, preprocessingParams)

N_beta = size(sino,3);

angle_degree = preprocessingParams.Delta_axis_tilt;

% rotation_method  = 'matlab';
rotation_method  = preprocessingParams.tiltCorrectionMethod

tic
if angle_degree~=0
	switch rotation_method
	case 'matlab'
		disp('Matlab rotation');
		for i = 1:N_beta
			sino(:,:,i) = imrotate( sino(:,:,i), angle_degree, 'bilinear', 'crop');
			wght(:,:,i) = imrotate( wght(:,:,i), angle_degree, 'bilinear', 'crop');
		end
		for i=1:size(jigMeasurementsSino,3) 
			jigMeasurementsSino(:,:,i) = imrotate( jigMeasurementsSino(:,:,i), angle_degree, 'bilinear', 'crop');
		end

	case 'shear'
		windowLen = 10;
		interp_method = 'linear';
		fprintf('Rotation by shearing: interpolation method %s, windowLen %d\n',interp_method,windowLen);

		for i = 1:N_beta
			sino(:,:,i) = rotate_by_shear( sino(:,:,i), angle_degree, windowLen, interp_method );
			wght(:,:,i) = rotate_by_shear( wght(:,:,i), angle_degree, windowLen, interp_method );
		end
		for i=1:size(jigMeasurementsSino,3) 
			jigMeasurementsSino(:,:,i) = rotate_by_shear( jigMeasurementsSino(:,:,i), angle_degree, windowLen, interp_method );
		end
	end
end
time_rot = toc;
fprintf('Rotation time: %d\n',time_rot );


return