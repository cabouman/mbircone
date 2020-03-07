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
if [ ! $# == 2 ]; then messageError "Wrong command line parameter"; generalError "$0 $@"; exit 1; fi

masterFile_listFile="inversion_input_arg.txt"
masterFile_relative=$(sed "$((3*${1}))!d" ${masterFile_listFile})
masterFile=$(readlink -f ${masterFile_relative})

estimate_gamma=${2}

if [[ ! -e ${masterFile} ]]; then messageError "masterFile does not exist: \"${masterFile}\"!"; generalError "$0 $@"; exit 1; fi
echo "Running with masterFile = \"${masterFile}\""

cd ../code/Inversion/run/
if [[  $? != 0 ]]; then messageError "cd to executable"; generalError "$0 $@"; exit 1; fi

./preprocessing_BHC.sh "${masterFile}" "${estimate_gamma}"
if [[  $? != 0 ]]; then messageError "exectuting ./preprocessing_BHC.sh"; generalError "$0 $@"; exit 1; fi




