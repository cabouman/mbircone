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
	>&2 echo "ERROR: Number of arguments wrong!"
	exit 1
fi

# Argumets: Try mode=help

mode="${1}"


# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

masterFile=$(readlink -f ../control/multiNode/master.txt)

if [[ ! -e ${masterFile} ]]; then messageError "masterFile \"${masterFile}\" does not exist!"; generalError "$0 $@"; exit 1; fi

../code/multiNode/./setup.sh "${masterFile}" $mode










