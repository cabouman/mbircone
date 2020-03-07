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



# runFolder: check and cd
runFolder=$(readlink -f "../code/Denoisers/qggmrf_4D/run/")
cd "${runFolder}"
if [[  $? != 0 ]]; then messageError "cd to runFolder: \"${runFolder}\"" ; generalError "$0 $@"; exit 1; fi


./makeall.sh
if [[  $? != 0 ]]; then messageError "executing: ./makeall.sh" ; generalError "$0 $@"; exit 1; fi

