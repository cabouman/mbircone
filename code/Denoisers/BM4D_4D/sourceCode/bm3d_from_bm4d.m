function [denoisedImg, sigma_est] = bm3d_from_bm4d(noisyImg, distribution, sigma, profile, do_wiener, verbose, searchWindowSize)

if size(noisyImg,3)~=1
	error('bm3d_from_bm4d: only accepts 2D images');
end

dummy = 0*noisyImg;
new_noisyImg = cat(3, noisyImg, dummy);

[new_denoisedImg, sigma_est] = bm4d(new_noisyImg, distribution, sigma, profile, do_wiener, verbose, searchWindowSize);

denoisedImg = new_denoisedImg(:,:,1);
