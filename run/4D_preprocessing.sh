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

master_4D=$(readlink -f "../control/Recon_4D/master.txt")

# runFolder: check and cd
runFolder=$(readlink -f "../code/Recon_4D/run/")
cd "${runFolder}"
if [[  $? != 0 ]]; then messageError "cd to runFolder: \"${runFolder}\"" ; generalError "$0 $@"; exit 1; fi

time ./run_preprocessing.sh ${master_4D}
if [[  $? != 0 ]]; then messageError "executing: ./run_preprocessing.sh" ; generalError "$0 $@"; exit 1; fi

