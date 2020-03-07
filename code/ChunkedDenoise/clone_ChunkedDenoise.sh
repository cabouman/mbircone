#!/usr/bin/env bash

generalError()
{
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
}


while getopts ":M:p:s:" option
do
    case $option in
        M) master=$(readlink -f "${OPTARG}");;
        p) prefix="${OPTARG}";;
        s) suffix="${OPTARG}";;
        ?)
            >&2 echo "    Unknown option -${option}!"           
            exit 1;;
    esac
done

cd $(dirname "$0")
cloneFile=$(readlink -f "../plainParams/Cloning/clone_single.sh")
if [[ ! -e ${cloneFile} ]]; then >&2 echo "cloneFile does not exist!"; generalError "$0 $@" ; exit 1; fi

#######

master_new=$(bash ${cloneFile} -M ${master} -c "deep" -p "${prefix}" -s "${suffix}" )
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

####### File names

bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "inputImageFName" > /dev/null
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "outputImageFName" > /dev/null
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "inputImageListList" > /dev/null
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "outputImageListList" > /dev/null
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "patchPositionList" > /dev/null
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "processIndexList" > /dev/null
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash ${cloneFile} -c "shallow" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "folderSuffix" > /dev/null
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi


#######
echo $master_new


