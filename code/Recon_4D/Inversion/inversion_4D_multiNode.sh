#!/usr/bin/env bash

generalError()
{
	>&2 echo "inversion 4D error"
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
}

messageError()
{
    >&2 echo "Error: \"${1}\""
}


if [[ $# -ne 3 ]]; then
	messageError "number of inputs not 3"
	generalError "$0 $@"
	exit 1
fi

multiNodeMaster=$(readlink -f ${1})
inversionScript=$(readlink -f ${2})
invMasterList=$(readlink -f ${3})

# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

plainParamsFile=$(readlink -f "../../plainParams/plainParams.sh")

if [[ ! -e ${multiNodeMaster} ]]; then messageError "multiNodeMaster does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${invMasterList} ]]; then messageError "invMasterList does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${inversionScript} ]]; then messageError "inversionScript does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${plainParamsFile} ]]; then messageError "plainParamsFile does not exist!"; generalError "$0 $@"; exit 1; fi


commandList="$(bash "${plainParamsFile}" -a get -m "${multiNodeMaster}" -F commandList -r)"
if [[  $? != 0 ]]; then messageError "resolving commandList" ; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${commandList} ]]; then messageError "commandList does not exist!"; generalError "$0 $@"; exit 1; fi



oldIFS="$IFS"; IFS=$'\n'; invMasterList_arr=($(cat ${invMasterList})); IFS="$oldIFS"
unset invMasterList_arr[0]


# setting up command list
> ${commandList}
for invMaster in "${invMasterList_arr[@]}"; do
	echo "${inversionScript} ${invMaster}" >> ${commandList}
done

../../multiNode/./multiNode.sh "${multiNodeMaster}"





