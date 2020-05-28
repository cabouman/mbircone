#!/usr/bin/env bash

generalError()
{
    >&2 echo "clone_Inversion error"
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
}

dir_backup=$(pwd)

while getopts ":M:p:s:m:" option
do
    case $option in
        M) master=$(readlink -f "${OPTARG}");;
        p) prefix="${OPTARG}";;
        s) suffix="${OPTARG}";;
        m) mode="${OPTARG}";;
        ?)
            >&2 echo "    Unknown option -${OPTARG}!"           
            exit 1;;
    esac
done

if [[ ! ${mode} == "deep" ]] && [[ ! ${mode} == "shallow" ]]; then
    >&2 echo "Unknown mode \"${mode}\""
    generalError "$0 $@"; exit 1
fi

cd $(dirname "$0")
BASEDIR=$(pwd)

cloneFile=$(readlink -f "../../plainParams/Cloning/clone_single.sh")
if [[ ! -e ${cloneFile} ]]; then >&2 echo "cloneFile ${cloneFile} does not exist!"; generalError "$0 $@" ; exit 1; fi

master_new=$(bash ${cloneFile} -M ${master} -c "deep" -p "${prefix}" -s "${suffix}" )
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash ${cloneFile} -c "deep" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "dataSetInfo" > /dev/null
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash ${cloneFile} -c "deep" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "preprocessingParams" > /dev/null
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash ${cloneFile} -c "deep" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" > /dev/null
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash ${cloneFile} -c "deep" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "sinoParams" > /dev/null
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash ${cloneFile} -c "deep" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "imgParams" > /dev/null
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash ${cloneFile} -c "deep" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "reconParams" > /dev/null
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash ${cloneFile} -c "deep" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "viewAngleList" > /dev/null
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi



if [[ ${mode} == "deep" ]]; then
    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "sino" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "driftSino" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "jigMeasurementsSino" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "origSino" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "wght" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "errSino" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "recon" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "reconROI" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "proxMapInput" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "lastChange" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "timeToChange" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "phantom" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "sysMatrix" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "wghtRecon" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "projInput" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "projOutput" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "estimateSino" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "consensusRecon" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "backprojlikeInput" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

    bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "binaryFNames" -f "backprojlikeOutput" > /dev/null
    if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi


fi

echo $master_new


cd $BASEDIR