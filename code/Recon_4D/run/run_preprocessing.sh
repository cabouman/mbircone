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

plainParamsFile=$(readlink -f "../../plainParams/plainParams.sh")
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

# Read from 4D master file
cd $(dirname "$master_4D")

# Read master list
invMasterList_rel=$(bash $plainParamsFile -a get -m $master_4D -F invMasterList)
invMasterList=$(readlink -f ${invMasterList_rel})
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

cd "${scriptDir}"


cd ../Preprocessing
./preprocessing_4D.sh ${invMasterList} 

cd "${scriptDir}"