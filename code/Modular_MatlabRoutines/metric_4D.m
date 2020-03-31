function statval = metric_4D( volume_fNameList1, volume_fNameList2 , type, mask, isLSfit)

if ~exist('type')
	type = 'rmse';
end
if ~exist('isLSfit')
	isLSfit = 0;
end

% computed mean squared error for two 4D volumes.
% can do mean squared value if volume_fNameList2 is empty

if isempty(volume_fNameList1) && isempty(volume_fNameList2)
	error('metric_4D: both file lists cannot be empty');
end

val = 0;

for volID=1:length(volume_fNameList1)

	vol1 = read3D(volume_fNameList1{volID}, 'float32');
	if isempty(volume_fNameList2)
		vol2 = 0*vol1;
	else 
		vol2 = read3D(volume_fNameList2{volID}, 'float32');
	end

	if ~exist('mask') 
		mask = ones(size(vol1));
	end
	if isempty(mask)
		mask = ones(size(vol1));
	end

	if isLSfit==1
		vol1 = LS_fit_vol(vol1, vol2, mask);
	end
	

	switch type
	case 'rmse'
		rmseval = mean((vol1(:).*mask(:)-vol2(:).*mask(:)).^2)/mean(mask(:));
		val = val + rmseval;

	case 'ssim'
		RangeMax = max(prctile(vol1(:),99),prctile(vol2(:),99));
		RangeMin = max(prctile(vol1(:),1),prctile(vol2(:),1));
		DynamicRange = RangeMax-RangeMin;
		[~,ssimmap] =  ssim(vol1, vol2, 'DynamicRange', DynamicRange);
		ssimval = mean(ssimmap(:).*mask(:))/mean(mask(:));
		val = val + ssimval;

	otherwise
		error('metric_4D: unrecognized type')

	end
end

val = val/length(volume_fNameList1);

statval = sqrt(val);


return
