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


# test input
if [[ $# -ne 1 ]]; then
	>&2 echo "ERROR: Number of arguments wrong!"
	>&2 echo "Use modes: 'thilo' or 'soumendu'"

	exit 1
fi


mode="${1}"


# cd to where the script is
executionDir=$(pwd)
cd "$(dirname $0)"
cd ..


if [[ "${mode}" = "thilo" ]]; then
	echo mode is thilo
	remoteDir="/scratch/brown/tbalke/coneBeam"
	userName="tbalke"
	remoteName="brown.rcac.purdue.edu"


elif [[ "${mode}" = "soumendu" ]]; then
	echo mode is soumendu
	remoteDir="/scratch/rice/s/smajee/cone_beam_project/Results/"
	userName="smajee"
	remoteName="rice.rcac.purdue.edu"

elif [[ "${mode}" = "souDepot" ]]; then
	echo mode is souDepot
	remoteDir="/depot/bouman/users/smajee/cone_beam_project/Results/"
	userName="smajee"
	remoteName="rice.rcac.purdue.edu"

elif [[ "${mode}" = "sm_test" ]]; then
	echo mode is sm_test
	remoteDir="/scratch/rice/s/smajee/cone_beam_project/Results_1/"
	userName="smajee"
	remoteName="rice.rcac.purdue.edu"

elif [[ "${mode}" = "smDepot" ]]; then
	echo mode is smDepot
	remoteDir="/depot/bouman/users/smajee/cone_beam_project/Results_1/"
	userName="smajee"
	remoteName="rice.rcac.purdue.edu"

elif [[ "${mode}" = "sm_gpuTrain" ]]; then
	echo mode is sm_gpuTrain
	remoteDir="/scratch/brown/smajee/cone_beam_project/Results/"
	userName="smajee"
	remoteName="brownGPU.rcac.purdue.edu"

else
	messageError "mode ${mode} unknown"
	generalError "$0 $@";
	exit 1
fi


echo "Remove everything except for ./binaries folder"
bash -c  "ssh ${remoteName} 'cd ${remoteDir}; find . -maxdepth 1 -not -path './binaries*' -not -path '.' -not -path '..' | xargs rm -rf'"
if [[  $? != 0 ]]; then messageError "Remove everything except for ./binaries folder" ; generalError "$0 $@"; exit 1; fi


echo "Upload everything except for ./binaries/, ./sftp-config.json, and ./.git/"
rsync -arv --quiet --exclude=binaries --exclude=sftp-config.json --exclude=.git . "${userName}@${remoteName}:${remoteDir}"
if [[  $? != 0 ]]; then messageError "Upload everything except for ./binaries/, ./sftp-config.json and ./.git/" ; generalError "$0 $@"; exit 1; fi





