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



# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

if [[ $# -ne 2 ]]; then
	messageError "number of inputs not 2"
	generalError "$0 $@"
	exit 1
fi
master_4D=$(readlink -f ${1})
mode=${2}

plainParamsFile=$(readlink -f "../../plainParams/plainParams.sh")
if [[  $? != 0 ]]; then messageError "reading plainparams" ; generalError "$0 $@"; exit 1; fi


cd "${scriptDir}"


cd ../Inversion
./inversion_4D_multiNode_smart.sh ${master_4D} ${mode}
if [[  $? != 0 ]]; then messageError "executing inversion_4D.sh" ; generalError "$0 $@"; exit 1; fi

cd "${scriptDir}"