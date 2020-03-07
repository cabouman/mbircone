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

# check arguments
if [ ! $# == 1 ]; then messageError "Wrong command line parameter (choose 1 or 2)"; generalError "$0 $@"; exit 1; fi

masterFile_listFile="inversion_input_arg.txt"
masterFile_relative=$(sed "$((3*${1}))!d" ${masterFile_listFile})
masterFile=$(readlink -f ${masterFile_relative})

if [[ ! -e ${masterFile} ]]; then messageError "masterFile does not exist: \"${masterFile}\"!"; generalError "$0 $@"; exit 1; fi
echo "Running with masterFile = \"${masterFile}\""

cd ../code/Inversion/run/
if [[  $? != 0 ]]; then messageError "cd to executable"; generalError "$0 $@"; exit 1; fi

time ./runall_exceptRecon.sh "${masterFile}"
if [[  $? != 0 ]]; then messageError "exectuting ./runall_exceptRecon.sh"; generalError "$0 $@"; exit 1; fi


