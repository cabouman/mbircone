function statval = stat_RMSE_4D( volume_fNameList1, volume_fNameList2 )

% computed mean squared error for two 4D volumes.
% can do mean squared value if volume_fNameList2 is empty

if isempty(volume_fNameList1) && isempty(volume_fNameList2)
	error('stat_RMSE_4D: both file lists cannot be empty');
end

val = 0;

for volID=1:length(volume_fNameList1)

	vol1 = read3D(volume_fNameList1{volID}, 'float32');
	if isempty(volume_fNameList2)
		vol2 = 0*vol1;
	else 
		vol2 = read3D(volume_fNameList2{volID}, 'float32');
	end

	val = val + immse_my(vol1, vol2);
end

val = val/length(volume_fNameList1);

statval = sqrt(val);


return