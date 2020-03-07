#!/usr/bin/env bash

generalError()
{
	>&2 echo "cloning 4D error"
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
}

messageError()
{
    >&2 echo "Error: \"${1}\""
}


if [[ $# -ne 2 ]]; then
	generalError "$0 $@"
	exit 1
fi

master_4D=$(readlink -f ${1})
master_inversion=$(readlink -f ${2})

cd $(dirname "$0")
BASEDIR=$(pwd)
plainParamsFile=$(readlink -f "../../plainParams/plainParams.sh")
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi
cloneFile=$(readlink -f "../../Inversion/run/clone_Inversion.sh")
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

# Read from 4D master file
cd $(dirname "$master_4D")

# Read params file for 4D
params_4D_rel=$(bash $plainParamsFile -a get -m $master_4D -F params_4D)
params_4D=$(readlink -f ${params_4D_rel})
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

# Read master list
invMasterList_rel=$(bash $plainParamsFile -a get -m $master_4D -F invMasterList)
invMasterList=$(readlink -f ${invMasterList_rel})
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

cd $BASEDIR

# Read param values
mode_4D=$(bash $plainParamsFile -a get -m $params_4D -F mode_4D)
num_timePoints=$(bash $plainParamsFile -a get -m $params_4D -F num_timePoints)
time_start=$(bash $plainParamsFile -a get -m $params_4D -F time_start)
time_end=$(bash $plainParamsFile -a get -m $params_4D -F time_end)
time_step=$(bash $plainParamsFile -a get -m $params_4D -F time_step)
num_viewSubsets=$(bash $plainParamsFile -a get -m $params_4D -F num_viewSubsets)
prefixRoot=$(bash $plainParamsFile -a get -m $params_4D -F prefixRoot)
suffixRoot=$(bash $plainParamsFile -a get -m $params_4D -F suffixRoot)


echo "mode_4D = $mode_4D"
echo "-  -  -  -  -  -  -  -  -  -  -  -  -  -  "
echo "num_timePoints = $num_timePoints"
echo "time_start = $time_start"
echo "time_end = $time_end"
echo "time_step = $time_step"
echo "-  -  -  -  -  -  -  -  -  -  -  -  -  -  "
echo "num_viewSubsets = $num_viewSubsets"
echo "-  -  -  -  -  -  -  -  -  -  -  -  -  -  "
echo "prefixRoot = $prefixRoot"
echo "suffixRoot = $suffixRoot"


# Adjusting to different modes
if [[ "${mode_4D}" = "view" ]]; then
	vol_id_start=0
	vol_id_end=$(($num_viewSubsets - 1))
	vol_id_step=1
	

elif [[ "${mode_4D}" = "time" ]]; then
	vol_id_start=$time_start
	vol_id_end=$time_end
	vol_id_step=$time_step
	

else
	messageError "mode_4D = \"${mode_4D}\" is unknown"
	generalError "$0 $@"
	exit 1
fi

# Computing and storing number of volumes
num_vols=$((0)) 
for (( vol_id=$vol_id_start; vol_id<=$vol_id_end; vol_id+=$vol_id_step ))
do
	num_vols=$(($num_vols + 1)) 
done
echo "-  -  -  -  -  -  -  -  -  -  -  -  -  -  "
echo "num_vols = $num_vols" 
echo "${num_vols}" > ${invMasterList}



for (( vol_id=$vol_id_start; vol_id<=$vol_id_end; vol_id+=$vol_id_step ))
do
	vol_str=$(printf %06d $vol_id)

	# clone
	suffix=${suffixRoot}
	prefix=${prefixRoot}${vol_str}"_"
	masterFile_temp=$(bash ${cloneFile} -M "${master_inversion}" -p "${prefix}" -s "${suffix}" -m deep )
	if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi
	echo $masterFile_temp

	echo "${masterFile_temp}" >> ${invMasterList}


	if [[ "${mode_4D}" = "view" ]]; then
		num_viewSubsets_clone=$num_viewSubsets
		index_viewSubsets_clone=$vol_id
		num_timePoints_clone=1
		index_timePoints_clone=0

	elif [[ "${mode_4D}" = "time" ]]; then
		num_viewSubsets_clone=1
		index_viewSubsets_clone=0
		num_timePoints_clone=$num_timePoints
		index_timePoints_clone=$vol_id
	fi

	$(bash $plainParamsFile -a set -m $masterFile_temp -F preprocessingParams -f "num_viewSubsets" -v $num_viewSubsets_clone )
	if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

	$(bash $plainParamsFile -a set -m $masterFile_temp -F preprocessingParams -f "index_viewSubsets" -v $index_viewSubsets_clone  )
	if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

	$(bash $plainParamsFile -a set -m $masterFile_temp -F preprocessingParams -f "num_timePoints" -v $num_timePoints_clone )
	if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

	$(bash $plainParamsFile -a set -m $masterFile_temp -F preprocessingParams -f "index_timePoints" -v $index_timePoints_clone  )
	if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi
done


