#!/usr/bin/env bash

generalError()
{
	>&2 echo "sino correction 4D error"
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

# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

invMasterList=$(readlink -f ${1})
option=${2}

echo $invMasterList

read -r num_timePoints < $invMasterList
echo "num_timePoints = $num_timePoints"

cd ../../Inversion/run
for (( t=0; t<$num_timePoints; t++ ))
do
	index=$(( t + 1 ))
	echo "Sino Correction at time: $index of $num_timePoints "
	line_num=$(( t + 2 ))
	masterFile_temp="`sed -n ${line_num}p $invMasterList`"
	echo "Sino Correction with masterfile: ${masterFile_temp}"

	if [[ ${option} = "correct_FOV" ]]; then
		echo "option: ${option}"
		./FOV_correct.sh $masterFile_temp "correct_FOV"
	fi

done

cd "${scriptDir}"

