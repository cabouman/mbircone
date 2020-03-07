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


master_4D=$(readlink -f "../control/Recon_4D/master.txt")
master_inversion_listFile="inversion_input_arg.txt"
master_inversion_relative=$(sed "$((3*${1}))!d" ${master_inversion_listFile})
master_inversion=$(readlink -f ${master_inversion_relative})
if [[  $? != 0 ]]; then messageError "Reading master_inversion"; generalError "$0 $@"; exit 1; fi

# runFolder: check and cd
runFolder=$(readlink -f "../code/Recon_4D/run/")
cd "${runFolder}"
if [[  $? != 0 ]]; then messageError "cd to runFolder: \"${runFolder}\"" ; generalError "$0 $@"; exit 1; fi

time ./run_cloning.sh ${master_4D} ${master_inversion}
if [[  $? != 0 ]]; then messageError "executing: ./run_cloning.sh" ; generalError "$0 $@"; exit 1; fi

