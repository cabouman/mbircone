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
if [[ $# -ne 2 ]]; then
	>&2 echo "ERROR: Input master file and method type!"
	exit 1
fi

masterFile=$(readlink -f ${1})
estimate_gamma=${2}

# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

plainParamsFile=$(readlink -f "../../plainParams/plainParams.sh")

if [[ ! -e ${masterFile} ]]; then messageError "masterFile does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${plainParamsFile} ]]; then messageError "plainParamsFile does not exist!"; generalError "$0 $@"; exit 1; fi

cd ../Preprocessing/BHC_polynomial/
if [[  $? != 0 ]]; then messageError "cd'ing to BHC_polynomial" ; generalError "$0 $@"; exit 1; fi

module load matlab
matlabSuccess=42; # has to be ~= 0
if [[ ${estimate_gamma} = "1" ]]; then
	echo "option: ${estimate_gamma}"
	echo "adjustBHCPolynomial_estimate"
	matlabCommand="adjustBHCPolynomial_estimate('${masterFile}', '${plainParamsFile}'); exit(${matlabSuccess});"
else
	echo "option: ${estimate_gamma}"
	echo "adjustBHCPolynomial"
	matlabCommand="adjustBHCPolynomial('${masterFile}', '${plainParamsFile}'); exit(${matlabSuccess});"
fi
matlab -nojvm -r "${matlabCommand}" < /dev/null
if [[  $? != ${matlabSuccess} ]]; then messageError "preprocessing failed. Command = \"${matlabCommand}\"" ; generalError "$0 $@"; exit 1; fi


