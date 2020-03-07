#!/usr/bin/env bash

generalError()
{
    >&2 echo "qggmrf/run/run.sh error"
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
}


if [[ $# -ne 2 ]]; then
	>&2 echo "ERROR: Input master file and plain params file!"
	exit 1
fi
masterFile=$(readlink -f ${1})
if [[ ! -e ${masterFile} ]]; then echo "masterFile does not exist!"; exit 1; fi
plainParamsFile=$(readlink -f ${2})
if [[ ! -e ${plainParamsFile} ]]; then echo "plainParamsFile does not exist!"; exit 1; fi

cd $(dirname ${0})


verbose_val=$(bash ${plainParamsFile} -a get -m ${masterFile} -F "params" -f "verbose")
#echo "verbose: $verbose_val"

# argument
arg="-a ${masterFile} -b ${plainParamsFile}"
#echo "arg = ${arg}"
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

if [[  $verbose_val != 1 ]]; then
	./go_reconstruct ${arg} > "denoising_output.txt"
else
	./go_reconstruct ${arg}
	# gdb --args go_reconstruct ${arg}
	# valgrind -v --leak-check=yes ./go_reconstruct ${arg}
fi
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

echo "Denoising Done on ${masterFile}"



