function x_s = CustomSegment3D(x, par)



if(strcmp(par.segmentationMethod, 'threshold'))
	%% -- Fixed Threshold --
	Threshold = par.segmentationThreshold;
	x_s = x;
	x_s(x< Threshold) = -Threshold;
	x_s(x>=Threshold) = Threshold;
	x_s = x_s / (2*Threshold) + 1/2;
end


if(strcmp(par.segmentationMethod, 'adaptive'))
	%% -- Adaptive Method --
	x_s1 = zeros(size(x));
	for i = 1:size(x,1)
	    x_s1(i,:,:) = reshape(imbinarize(squeeze(x(i,:,:))), size(x(i,:,:)));
	end

	x_s2 = zeros(size(x));
	for i = 1:size(x,2)
	    x_s2(:,i,:) = reshape(imbinarize(squeeze(x(:,i,:))), size(x(:,i,:)));
	end

	x_s3 = zeros(size(x));
	for i = 1:size(x,3)
	    x_s3(:,:,i) = reshape(imbinarize(squeeze(x(:,:,i))), size(x(:,:,i)));
	end

	x_s = x_s1 .* x_s2 .* x_s3;
end


% Solve LS || x_s al - x||^2_W for al
[al] = solveWeightedLS(x_s(:), x(:), ones(size(x(:))));
x_s = x_s * al;




end