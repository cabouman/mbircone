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


nodeName="${1}"
commandList="${2}"
id_command_start="${3}"
id_command_stop="${4}"
numAgentsPerNode="${5}"

# echo nodeName = $nodeName
# echo commandList = $commandList
# echo id_command_start = $id_command_start
# echo id_command_stop = $id_command_stop
# echo numAgentsPerNode = $numAgentsPerNode

verbosity=0

if [[ $verbosity = 0 ]]; then
	verbosityFlag=""
else
	verbosityFlag="-t"
fi

oldIFS="$IFS"; IFS=$'\n'; commandList_arr=($(<${commandList})); IFS="$oldIFS"
commandList_len=${#commandList_arr[@]}

numCommands=$(( $id_command_stop - $id_command_start + 1 ))


printf '%s\n' "${commandList_arr[@]:${id_command_start}:${numCommands}}" | xargs ${verbosityFlag} -P $numAgentsPerNode -I % bash -c "ssh -t -t ${nodeName} 'bash -O huponexit -c \"source /etc/profile; echo Starting on node: \$(hostname); %; echo Done on node: \$(hostname)\"'"
if [[  $? != 0 ]]; then messageError "Error on node: ${nodeName}" ; generalError "$0 $@"; exit 1; fi


