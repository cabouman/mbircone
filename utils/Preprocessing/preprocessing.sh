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

plainParamsFile=$(readlink -f "../plainParams/plainParams.sh")

if [[ ! -e ${masterFile} ]]; then messageError "masterFile does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${plainParamsFile} ]]; then messageError "plainParamsFile does not exist!"; generalError "$0 $@"; exit 1; fi

preprocessingScript="$(bash ${plainParamsFile} -a get -m ${masterFile} -F preprocessingScript -r)"
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi
if [[ ! -e ${preprocessingScript} ]]; then messageError "preprocessingScript does not exist!"; generalError "$0 $@"; exit 1; fi

bash "${preprocessingScript}" "${masterFile}"

applyJigCorrectionMode=$(bash "${plainParamsFile}" -a get -m "${masterFile}" -F preprocessingParams -f applyJigCorrectionMode)
if [[  $? != 0 ]]; then messageError "error reading applyJigCorrectionMode field!"; generalError "$0 $@"; exit 1; fi
JigCorrectionScript=$(readlink -f "../JigCorrection/./jig_correction_apply.sh");

if [[ "${applyJigCorrectionMode}" != '0' ]]; then
	bash "${JigCorrectionScript}" "${masterFile}" "${plainParamsFile}"
	if [[  $? != 0 ]]; then messageError "error running JigCorrectionScript!"; generalError "$0 $@"; exit 1; fi
fi
