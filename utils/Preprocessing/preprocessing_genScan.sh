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


# test input
if [[ $# -ne 1 ]]; then
    >&2 echo "ERROR: Input master file!"
    exit 1
fi

masterFile=$(readlink -f ${1})



# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

preprocessingScript="$(readlink -f ../Preprocessing/genScan_readScans/readScans_and_preprocess_genScan.m)"
plainParamsFile=$(readlink -f "../plainParams/plainParams.sh")

if [[ ! -e ${masterFile} ]]; then messageError "masterFile does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${plainParamsFile} ]]; then messageError "plainParamsFile does not exist!"; generalError "$0 $@"; exit 1; fi


preprocessingScript_dir=$(dirname ${preprocessingScript})

fName_temp=$(basename $preprocessingScript)
preprocessingFunction="${fName_temp%.*}"


if [[ ! -e ${preprocessingScript_dir} ]]; then messageError "preprocessingScript_dir does not exist!"; generalError "$0 $@"; exit 1; fi
cd "${preprocessingScript_dir}"

module load matlab
matlabSuccess=42; # has to be ~= 0
matlabCommand="${preprocessingFunction}('${masterFile}', '${plainParamsFile}'); exit(${matlabSuccess});"
matlab -nojvm -r "${matlabCommand}" < /dev/null
if [[  $? != ${matlabSuccess} ]]; then messageError "preprocessing failed. Command = \"${matlabCommand}\"" ; generalError "$0 $@"; exit 1; fi


