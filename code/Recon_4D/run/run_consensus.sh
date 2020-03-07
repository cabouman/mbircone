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
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

master_4D=$(readlink -f ${1})

cd ../Consensus
./run.sh ${master_4D} 

cd "${scriptDir}"