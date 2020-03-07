#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
	>&2 echo "ERROR: Input master file!"
	exit 1
fi
masterFile=$(readlink -f ${1})

cd $(dirname ${0})
plainParamsFile_rel="../../../plainParams/plainParams.sh"
plainParamsFile=$(readlink -f ${plainParamsFile_rel})

echo $plainParamsFile

module load matlab
matlabSuccess=1
matlab -nosplash -nojvm -r "genData_run('${masterFile}', '${plainParamsFile}'); exit(${matlabSuccess});" </dev/null
if [[  $? != ${matlabSuccess} ]]; then >&2 echo "Error in rungen.sh"; exit 1; fi


