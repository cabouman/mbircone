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

usage()
{
    messageError "Unknown Mode"
    echo "modes:"
    echo "            init:                 Clears and then saves node info"
    echo "            append:               Only saves node info"
    echo "            check:                Prints masterFile"
    echo "            check_all:            Prints masterFile and hostNameList"
    echo "            login:                ssh to first hostname in the list"
    exit 1
}

# test input
if [[ $# -ne 2 ]]; then
    >&2 echo "ERROR: Number of arguments wrong!"
    usage
    exit 1
fi

masterFile=$(readlink -f ${1})
mode="${2}"


# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

plainParamsFile=$(readlink -f "../plainParams/plainParams.sh")

if [[ ! -e ${masterFile} ]]; then messageError "masterFile \"${masterFile}\" does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${plainParamsFile} ]]; then messageError "plainParamsFile \"${plainParamsFile}\" does not exist!"; generalError "$0 $@"; exit 1; fi

#echo EXIT; exit 0

if [[ ${mode} = "init" || ${mode} = "append" ]]; then


        bash ${plainParamsFile} -a set -m ${masterFile} -F numCoresPerNode -v ${PBS_NUM_PPN}
        if [[  $? != 0 ]]; then messageError "Error accessing masterFile" ; generalError "$0 $@"; exit 1; fi

        bash ${plainParamsFile} -a set -m ${masterFile} -F date -v "$(date)"
        if [[  $? != 0 ]]; then messageError "Error accessing masterFile" ; generalError "$0 $@"; exit 1; fi

        hostNameList=$(bash ${plainParamsFile} -a get -m ${masterFile} -F hostNameList -r)
        if [[  $? != 0 ]]; then messageError "Error accessing masterFile" ; generalError "$0 $@"; exit 1; fi

        if [[ ${mode} = "init" ]]; then
            cat "${PBS_NODEFILE}" > "${hostNameList}"
        fi
        if [[ ${mode} = "append" ]]; then
            cat "${PBS_NODEFILE}" >> "${hostNameList}"
        fi

elif [[ ${mode} = "check" ]]; then
    
        hostNameList=$(bash ${plainParamsFile} -a get -m ${masterFile} -F hostNameList -r)
        if [[  $? != 0 ]]; then messageError "Error accessing masterFile" ; generalError "$0 $@"; exit 1; fi

        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        echo "~~~~~~~~  master:  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        cat "${masterFile}"
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

elif [[ ${mode} = "check_all" ]]; then
    
        hostNameList=$(bash ${plainParamsFile} -a get -m ${masterFile} -F hostNameList -r)
        if [[  $? != 0 ]]; then messageError "Error accessing masterFile" ; generalError "$0 $@"; exit 1; fi

        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        echo "~~~~~~~~  master:  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        cat "${masterFile}"
        echo
        echo "~~~~~~~~  nodes:   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        cat "${hostNameList}"
        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

elif [[ ${mode} = "login" ]]; then

        hostNameList=$(bash ${plainParamsFile} -a get -m ${masterFile} -F hostNameList -r)
        if [[  $? != 0 ]]; then messageError "Error accessing masterFile" ; generalError "$0 $@"; exit 1; fi

        bash -c "ssh $(sed '1q;d' ${hostNameList})"
        if [[  $? != 0 ]]; then messageError "Executing bash -c \"ssh $(sed '1q;d' ${hostNameList})\"" ; generalError "$0 $@"; exit 1; fi

else
        usage
        generalError "$0 $@"
        exit 1
fi











