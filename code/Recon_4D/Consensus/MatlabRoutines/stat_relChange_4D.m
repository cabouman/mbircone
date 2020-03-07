function statval = stat_relChange_4D( volume_fNameList1, volume_fNameList2 )

% computed relative change for two 4D volumes.

meanChange = 0;
meanVal = 0;


for volID=1:length(volume_fNameList1)

	vol1 = read3D(volume_fNameList1{volID}, 'float32');
	vol2 = read3D(volume_fNameList2{volID}, 'float32');
    
    value = max(vol1, vol2);
    avg_value = mean(value(:));

    change = abs(vol1(:) - vol2(:));
    avg_chage = mean(change(:));

    meanChange = meanChange + avg_chage;
    meanVal = meanVal + avg_value;

end

statval = meanChange / meanVal * 100;

return 
