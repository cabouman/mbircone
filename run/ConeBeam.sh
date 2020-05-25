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
    >&2 echo "Usage:"
    >&2 echo "argument 1:  <masterFile>"    
    >&2 echo "argument 2:  help"    
    >&2 echo "             <task>"
    >&2 echo "                          CBMODE_preprocessing"
    >&2 echo "                          CBMODE_changePreprocessing"
    >&2 echo "                          CBMODE_multiResolution"
    >&2 echo "                          CBMODE_FDK"
    >&2 echo "                          CBMODE_INV_sys"
    >&2 echo "                          CBMODE_INV_wghtRecon"
    >&2 echo "                          CBMODE_INV_init"
    >&2 echo "                          CBMODE_INV_recon"
    >&2 echo "                          CBMODE_INV_runall"
    >&2 echo "                          CBMODE_INV_prepare"
    >&2 echo "                          CBMODE_proj"
    >&2 echo "                          CBMODE_backprojlike"
    >&2 echo "                          help"
    >&2 echo ""
}

FDK_if_initReconMode_FDK()
{
    initReconMode=$(bash "${plainParamsFile}" -a get -m "${masterFile}" -F reconParams -f initReconMode)
    echo "initReconMode: $initReconMode"
    if [[ "${initReconMode}" = "FDK" ]]; then

        ./FDK/./FDK.sh "${masterFile}"
        if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    fi
}

# test input
if [[ $# -ne 2 ]]; then
    >&2 echo "ERROR: Number of arguments incorrect!"
    usage
    exit 1
fi

masterFile=$(readlink -f ${1})
if [[ ! -e ${masterFile} ]]; then messageError "masterFile \"${masterFile}\" does not exist!"; generalError "$0 $@"; exit 1; fi

task=${2}


# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

plainParamsFile=$(readlink -f "../utils/plainParams/plainParams.sh")
if [[ ! -e ${plainParamsFile} ]]; then messageError "plainParamsFile \"${plainParamsFile}\" does not exist!"; generalError "$0 $@"; exit 1; fi

echo task = $task



# -------- Other tasks --------------------------------------------
if [[ "${task}" = "CBMODE_preprocessing" ]]; then

    bash ../utils/Preprocessing/./preprocessing.sh "${masterFile}"
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

elif [[ "${task}" = "CBMODE_changePreprocessing" ]]; then

    bash ../utils/multiResolution/./changePreprocessing.sh "${masterFile}"
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

elif [[ "${task}" = "CBMODE_multiResolution" ]]; then

    bash ../utils/multiResolution/./multiResolution.sh "${masterFile}"
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

elif [[ "${task}" = "CBMODE_FDK" ]]; then

    bash ../utils/FDK/./FDK.sh "${masterFile}"
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

# -------- Core Inversion --------------------------------------------
elif [[ "${task}" = "CBMODE_INV_sys" ]]; then

    bash ./run.sh "${masterFile}" sys
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

elif [[ "${task}" = "CBMODE_INV_wghtRecon" ]]; then

    bash ./run.sh "${masterFile}" wghtRecon
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

elif [[ "${task}" = "CBMODE_INV_init" ]]; then

    FDK_if_initReconMode_FDK
    bash ./run.sh "${masterFile}" init
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

elif [[ "${task}" = "CBMODE_INV_recon" ]]; then

    bash ./run.sh "${masterFile}" recon
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

elif [[ "${task}" = "CBMODE_INV_runall" ]]; then

    FDK_if_initReconMode_FDK
    bash ./run.sh "${masterFile}" sys,wghtRecon,init,recon
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

elif [[ "${task}" = "CBMODE_INV_prepare" ]]; then

    FDK_if_initReconMode_FDK
    bash ./run.sh "${masterFile}" sys,wghtRecon,init
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

elif [[ "${task}" = "CBMODE_proj" ]]; then

    bash ./run.sh "${masterFile}" proj
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

elif [[ "${task}" = "CBMODE_backprojlike" ]]; then

    bash ./run.sh "${masterFile}" backprojlike
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi


# -------- Help --------------------------------------------

elif [[ "${task}" = "help" ]]; then

    usage

# -------- Error --------------------------------------------
else

    messageError "Mode ${task} is unknown"
    generalError "$0 $@"
    usage
    exit 1
fi
