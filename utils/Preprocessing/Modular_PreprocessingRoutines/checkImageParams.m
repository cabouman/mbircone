function [] = checkImageParams(data)

if(data.parImg.j_zstart_roi > data.parImg.j_zstop_roi)
	disp('Sino Params:')
	data.par
	disp('Image Params:')
	data.parImg
	error('Problem with image params: ROI volume is negative');
end

end
