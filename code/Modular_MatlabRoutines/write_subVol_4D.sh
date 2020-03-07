#!/usr/bin/env bash

generalError()
{
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
}

messageError()
{
    >&2 echo "Error: \"${1}\""
}

if [ ! $# == 2 ]; then messageError "Wrong number of command line parameters"; generalError "$0 $@"; exit 1; fi

fNameList_3D_fName=$(readlink -f ${1})
fName_timeVol=$(readlink -f ${2})

echo $fNameList_3D_fName
echo $fName_timeVol

# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"


module load matlab
matlabSuccess=42; # has to be ~= 0
matlabCommand="write_subVolume_4D_staticPaths('${fNameList_3D_fName}', '${fName_timeVol}'); exit(${matlabSuccess});"
matlab -nojvm -r "${matlabCommand}" < /dev/null
if [[  $? != ${matlabSuccess} ]]; then messageError "write_subVolume_4D_staticPaths failed. Command = \"${matlabCommand}\"" ; generalError "$0 $@"; exit 1; fi

cd "${scriptDir}"
