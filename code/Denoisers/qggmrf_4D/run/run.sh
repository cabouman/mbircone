#!/usr/bin/env bash

generalError()
{
    >&2 echo "qggmrf/run/run.sh error"
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
}


if [[ $# -ne 1 ]]; then
	>&2 echo "ERROR: Input master file!"
	exit 1
fi
masterFile=$(readlink -f ${1})
if [[ ! -e ${masterFile} ]]; then echo "masterFile does not exist!"; exit 1; fi


cd $(dirname ${0})

# for debug
# ./makeall.sh

plainParamsFile=$(readlink -f "../../../plainParams/plainParams.sh")

cd ../C_Code/Recon/
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi
./runRecon.sh ${masterFile} ${plainParamsFile}
if [[  $? != 0 ]]; then >&2 echo "Error running runRecon.sh" ; exit 1; fi




