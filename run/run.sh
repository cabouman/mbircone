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

usage()
{
    >&2 echo "Usage:"
    >&2 echo "argument:    help"    
    >&2 echo "             all"
    >&2 echo "             clean"  

}



# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"
echo $(pwd)
if [[ $# -ne 2 ]]; then
    messageError "Number of arguments incorrect"
    generalError "$0 $@"
    usage
    exit 1
fi

masterFile=${1}
modes=${2}

plainParamsFile=$(readlink -f "../utils/plainParams/plainParams.sh")
if [[ ! -e ${plainParamsFile} ]]; then messageError "plainParamsFile \"${plainParamsFile}\" does not exist!"; generalError "$0 $@"; exit 1; fi


#cd source/
# ----------------------------------------------------------------------------------------- sys,init,recon
../bin/./main -a "${masterFile}" -b "${plainParamsFile}" -c "${modes}"
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi
