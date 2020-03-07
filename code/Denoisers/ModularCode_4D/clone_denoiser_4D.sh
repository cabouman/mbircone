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
cloneFile=$(readlink -f "../../plainParams/Cloning/clone_single.sh")
cloneListFile=$(readlink -f "../../plainParams/Cloning/clone_fileList.sh")
plainParamsFile=$(readlink -f "../../plainParams/plainParams.sh")
if [[ ! -e ${cloneFile} ]]; then >&2 echo "cloneFile does not exist!"; generalError "$0 $@" ; exit 1; fi

#######

master_new=$(bash ${cloneFile} -M ${master} -c "deep" -p "${prefix}" -s "${suffix}" )
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

#######

# bash ${cloneFile} -c "deep" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "params" > /dev/null
# if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

####### File lists file

bash ${cloneFile} -c "deep" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "noisyBinaryFName_timeList" > /dev/null
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash ${cloneFile} -c "deep" -p "${prefix}" -s "${suffix}" -M "${master_new}" -F "denoisedBinaryFName_timeList" > /dev/null
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi


####### File lists
noisyBinaryFName_timeList_new=$(bash $plainParamsFile -a get -m $master_new -F "noisyBinaryFName_timeList" -r)
denoisedBinaryFName_timeList_new=$(bash $plainParamsFile -a get -m $master_new -F "denoisedBinaryFName_timeList" -r)

# echo "noisyBinaryFName_timeList_new = $noisyBinaryFName_timeList_new"
# echo "denoisedBinaryFName_timeList_new = $denoisedBinaryFName_timeList_new"

bash ${cloneListFile} -c "shallow" -p "${prefix}" -s "${suffix}" -F "${noisyBinaryFName_timeList_new}"
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

bash ${cloneListFile} -c "shallow" -p "${prefix}" -s "${suffix}" -F "${denoisedBinaryFName_timeList_new}"
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

#######
echo $master_new

