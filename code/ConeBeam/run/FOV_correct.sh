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

masterFile=$(readlink -f ${1})
option=${2}

# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

plainParamsFile=$(readlink -f "../../plainParams/plainParams.sh")
if [[ ! -e ${masterFile} ]]; then messageError "masterFile does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${plainParamsFile} ]]; then messageError "plainParamsFile does not exist!"; generalError "$0 $@"; exit 1; fi

cd ../Preprocessing/FOV_correction/
if [[  $? != 0 ]]; then messageError "cd'ing to FOV_correction" ; generalError "$0 $@"; exit 1; fi

module load matlab
matlabSuccess=42; # has to be ~= 0
if [[ ${option} = "maskProject" ]]; then
	echo "option: ${option}"
	matlabCommand="maskProject('${masterFile}', '${plainParamsFile}', /FOV_correction/); exit(${matlabSuccess});"
fi

if [[ ${option} = "maskSino" ]]; then
	echo "option: ${option}"
	matlabCommand="maskSino('${masterFile}', '${plainParamsFile}'); exit(${matlabSuccess});"
fi

if [[ ${option} = "correct_FOV" ]]; then
	echo "option: ${option}"
	matlabCommand="correct_FOV('${masterFile}', '${plainParamsFile}'); exit(${matlabSuccess});"
fi

matlab -nojvm -r "${matlabCommand}" < /dev/null
if [[  $? != ${matlabSuccess} ]]; then messageError "run failed. Command = \"${matlabCommand}\"" ; generalError "$0 $@"; exit 1; fi


