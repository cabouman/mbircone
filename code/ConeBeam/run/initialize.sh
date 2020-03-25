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

plainParamsFile=$(readlink -f "../../plainParams/plainParams.sh")

if [[ ! -e ${masterFile} ]]; then messageError "masterFile \"${masterFile}\" does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${plainParamsFile} ]]; then messageError "plainParamsFile \"${plainParamsFile}\" does not exist!"; generalError "$0 $@"; exit 1; fi

arg="-a ${masterFile} -b ${plainParamsFile}"

echo ""
echo "###########################################################"
echo "                Run Initialization"
echo "###########################################################"

cd ../Code/0C_Initialize/
if [[  $? != 0 ]]; then messageError "executing \"cd 0B_GenSysMatrix/\"" ; generalError "$0 $@"; exit 1; fi
./initialize ${arg}
if [[  $? != 0 ]]; then messageError "executing ./initialize" ; generalError "$0 $@"; exit 1; fi
cd "${scriptDir}"



