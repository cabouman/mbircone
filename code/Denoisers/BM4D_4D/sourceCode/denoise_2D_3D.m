function [x_est] = denoise_2D_3D(x, modeStr, par)

mean_val = mean(x(:));
if sum(x(x(:)~=mean_val)) == 0
	% constant image: do nothing: bm4d breaks here
	x_est = x;
else
	switch modeStr
	case '2D'
	[x_est, sigma_est] = bm3d_from_bm4d(x, par.distribution, (~par.estimate_sigma)*par.sigma, par.profile, par.do_wiener, par.verbose, par.searchWindowSize);
		
	case '3D'
	[x_est, sigma_est] = bm4d(x, par.distribution, (~par.estimate_sigma)*par.sigma, par.profile, par.do_wiener, par.verbose, par.searchWindowSize);

	otherwise
	error('denoise_2D_3D: unknown mode')
	
end

if(par.estimate_sigma)
	disp(['Estimated sigma = ', num2str(mean(sigma_est(:)))]);
end


end