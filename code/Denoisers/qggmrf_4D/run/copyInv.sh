#!/usr/bin/env bash

noisyList=$(readlink -f ../../../../control/Denoisers/qggmrf_4D/noisyBinaryFName_timeList.txt)
denoisedList=$(readlink -f ../../../../control/Denoisers/qggmrf_4D/denoisedBinaryFName_timeList.txt)
T_num=4
echo $T_num > $noisyList
echo $T_num > $denoisedList
id=000030

time_start=30
time_end=33
time_step=1
for (( t_id=$time_start; t_id<=$time_end; t_id+=$time_step ))
do
	timeStr=$(printf %06d $t_id)
	t=$(($t_id - 1)) 

	fileInv=../../../../binaries/Inversion/timeclone_T_${timeStr}_object.recon
	fileInv_abs=$(readlink -f ${fileInv})
	echo $fileInv_abs

	fileNoisy=../../../../binaries/qggmrf/timeclone_T_${timeStr}_noisyImage.recon
	fileNoisy_abs=$(readlink -f ${fileNoisy})
	echo $fileNoisy_abs >> $noisyList
	echo $fileNoisy_abs

	fileDenoised=../../../../binaries/qggmrf/timeclone_T_${timeStr}_denoisedImage.recon
	fileDenoised_abs=$(readlink -f ${fileDenoised})
	echo $fileDenoised_abs >> $denoisedList
	echo $fileDenoised_abs
	
	cp -r $fileInv_abs $fileNoisy_abs
	cp -r $fileInv_abs $fileDenoised_abs

done