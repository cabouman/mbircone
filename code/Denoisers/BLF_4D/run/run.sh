#!/usr/bin/env bash

generalError()
{
    >&2 echo "BLF_4D/run/run.sh error"
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
}


if [[ $# -ne 1 ]]; then
	>&2 echo "ERROR: Input master file!"
	exit 1
fi
masterFile=$(readlink -f ${1})
if [[ ! -e ${masterFile} ]]; then echo "masterFile does not exist!"; exit 1; fi


cd $(dirname ${0})


plainParamsFile=$(readlink -f "../../../plainParams/plainParams.sh")

cd ../sourceCode/


module load matlab
matlabSuccess=1
matlab -nosplash -nojvm -r "denoise('${masterFile}', '${plainParamsFile}'); exit(${matlabSuccess});" </dev/null
if [[  $? != ${matlabSuccess} ]]; then >&2 echo "Error in denoise"; exit 1; fi


# echo "denoise('${masterFile}', '${plainParamsFile}');"
# matlab -nosplash -nojvm -r "denoise('${masterFile}', '${plainParamsFile}');"

