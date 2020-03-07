#!/usr/bin/env bash

generalError()
{
	>&2 echo "preprocessing 4D error"
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
}

messageError()
{
    >&2 echo "Error: \"${1}\""
}


if [[ $# -ne 1 ]]; then
	generalError "$0 $@"
	exit 1
fi

# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

invMasterList=$(readlink -f ${1})

echo $invMasterList

read -r num_vols < $invMasterList
echo "num_vols = $num_vols"

cd ../../Inversion/run
for (( vol_id=1; vol_id<=$num_vols; vol_id++ ))
do
	echo "Preprocessing volume: $vol_id of $num_vols "
	line_num=$(( vol_id + 1 ))
	masterFile_temp="`sed -n ${line_num}p $invMasterList`"
	echo "masterfile: ${masterFile_temp}"

	./preprocessing.sh $masterFile_temp
done

cd "${scriptDir}"

