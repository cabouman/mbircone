#!/usr/bin/env bash

generalError()
{
	>&2 echo "inversion 4D error"
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
}

messageError()
{
    >&2 echo "Error: \"${1}\""
}


if [[ $# -ne 2 ]]; then
	messageError "number of inputs not 2"
	generalError "$0 $@"
	exit 1
fi

master_4D=$(readlink -f ${1})
mode=${2}

# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

plainParamsFile=$(readlink -f "../../plainParams/plainParams.sh")


invMasterList=$(bash $plainParamsFile -a get -m $master_4D -F invMasterList -r)
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

multiNodeMaster=$(bash $plainParamsFile -a get -m $master_4D -F params_4D -f multiNodeMaster -r)
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

numAgentsPerNode=$(bash $plainParamsFile -a get -m $master_4D -F params_4D -f numAgentsPerNode )
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi


if [[ ! -e ${multiNodeMaster} ]]; then messageError "multiNodeMaster does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${invMasterList} ]]; then messageError "invMasterList does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${plainParamsFile} ]]; then messageError "plainParamsFile does not exist!"; generalError "$0 $@"; exit 1; fi


# set numAgentsPerNode inside multiNodeMaster
$(bash $plainParamsFile -a set -m $multiNodeMaster -F numAgentsPerNode -v $numAgentsPerNode )


read -r num_vols < $invMasterList
echo "num_vols = $num_vols"
echo "mode: $mode"

if [[ ${mode} = "runall" ]]; then
	script_3D=$(readlink -f "../../Inversion/run/runall.sh")
elif [[ ${mode} = "Recon" ]]; then

	script_3D=$(readlink -f "../../Inversion/run/Recon.sh")

elif [[ ${mode} = "preprocessing" ]]; then

	script_3D=$(readlink -f "../../Inversion/run/preprocessing.sh")

elif [[ ${mode} = "genSysMatrix" ]]; then

	script_3D=$(readlink -f "../../Inversion/run/genSysMatrix.sh")

elif [[ ${mode} = "initialize" ]]; then

	script_3D=$(readlink -f "../../Inversion/run/initialize.sh")

elif [[ ${mode} = "runall_exceptRecon" ]]; then

	script_3D=$(readlink -f "../../Inversion/run/runall_exceptRecon.sh")

elif [[ ${mode} = "multiResolution" ]]; then

	script_3D=$(readlink -f "../../Inversion/run/multiResolution.sh")

elif [[ ${mode} = "changePreprocessing" ]]; then

	script_3D=$(readlink -f "../../Inversion/run/changePreprocessing.sh")
else
	messageError "mode unknown"; generalError "$0 $@"; exit 1
fi


./inversion_4D_multiNode.sh ${multiNodeMaster} ${script_3D} ${invMasterList}


cd "${scriptDir}"

