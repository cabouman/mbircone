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

master_4D=$(readlink -f ${1})
master_inversion=$(readlink -f ${2})

cd ../Cloning
./cloning_4D.sh ${master_4D} ${master_inversion}

cd "${scriptDir}"