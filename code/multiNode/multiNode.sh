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

listDescendents ()
{
  local children=$(ps -o pid= --ppid "$1")

  for pid in $children
  do
    listDescendents "$pid"
  done

  echo "$children"
}

killAllDescendents()
{
	local process_list="$(listDescendents $$)"
	local process
	for process in $process_list; do

		# for all processes that still exist: kill them
		if ps -p $process_list > /dev/null
		then
			kill $process_list
		fi

	done
}


# test input
if [[ $# -ne 1 ]]; then
	>&2 echo "ERROR: Input master file!"
	exit 1
fi

masterFile=$(readlink -f ${1})

# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

plainParamsFile=$(readlink -f "../plainParams/plainParams.sh")


if [[ ! -e ${masterFile} ]]; then messageError "masterFile does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${plainParamsFile} ]]; then messageError "plainParamsFile does not exist!"; generalError "$0 $@"; exit 1; fi




# Reading Master File
hostNameList="$(bash "${plainParamsFile}" -a get -m "${masterFile}" -F hostNameList -r)"
if [[  $? != 0 ]]; then messageError "resolving hostNameList" ; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${hostNameList} ]]; then messageError "hostNameList does not exist!"; generalError "$0 $@"; exit 1; fi

commandList="$(bash "${plainParamsFile}" -a get -m "${masterFile}" -F commandList -r)"
if [[  $? != 0 ]]; then messageError "resolving commandList" ; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${commandList} ]]; then messageError "commandList does not exist!"; generalError "$0 $@"; exit 1; fi

numCoresPerNode="$(bash "${plainParamsFile}" -a get -m "${masterFile}" -F numCoresPerNode)"
numAgentsPerNode="$(bash "${plainParamsFile}" -a get -m "${masterFile}" -F numAgentsPerNode)"


# print master file
echo -------------------------------------
echo master file contents:
echo -------------------------------------
echo hostNameList = $hostNameList
echo commandList = $commandList
echo numCoresPerNode = $numCoresPerNode
echo numAgentsPerNode = $numAgentsPerNode
echo -------------------------------------




oldIFS="$IFS"; IFS=$'\n'; commandList_arr=($(<${commandList})); IFS="$oldIFS"
commandList_len=${#commandList_arr[@]}

oldIFS="$IFS"; IFS=$'\n'; hostNameList_arr=($(<${hostNameList})); IFS="$oldIFS"
hostNameList_len=${#hostNameList_arr[@]}

num_Nodes=$(($hostNameList_len / $numCoresPerNode))



echo "--------- multiNode: Execution   -------------------"

# This command is executed when this script is being killed by one of the listed signals
# It will find all the processes that have been created recursively and then kills them.
trap '$(killAllDescendents)' SIGINT SIGKILL SIGTERM SIGSTOP 

pids=""
for (( id_nodes = 0; id_nodes < $num_Nodes; id_nodes++ )); do
	
	id_command_start=$(( $id_nodes * $commandList_len / $num_Nodes))
	id_command_stop=$(( ($id_nodes + 1) * $commandList_len / $num_Nodes - 1))
	id_hostName=$(($id_nodes * $numCoresPerNode))
	nodeName="${hostNameList_arr[$id_hostName]}"
	#echo --------- multiNode: -------------------
	echo Running on Node $id_nodes: $nodeName
	#echo ----------------------------------------
	# echo id_hostName = $id_hostName
	# echo nodeName = $nodeName
	# echo id_command_start = $id_command_start
	# echo id_command_stop = $id_command_stop

	./singleNode.sh "${nodeName}" "${commandList}" "${id_command_start}" "${id_command_stop}" "${numAgentsPerNode}" &
	pids+=" $!"

done
pids_arr=($pids)


status_of_pids=""
for (( id_nodes = 0; id_nodes < $num_Nodes; id_nodes++ )); do
	
	p=${pids_arr[$id_nodes]}
	wait $p
	status="$?"
	status_of_pids+=" $status"

done
status_of_pids_arr=($status_of_pids)


total_status=0
echo "--------- multiNode: Exit Status -------------------"
for (( id_nodes = 0; id_nodes < $num_Nodes; id_nodes++ )); do
	
	pid=${pids_arr[$id_nodes]}
	status="${status_of_pids_arr[$id_nodes]}"

	id_hostName=$(($id_nodes * $numCoresPerNode))
	nodeName="${hostNameList_arr[$id_hostName]}"


	if [[ $status = 0 ]]; then
		echo "Process $pid success (node: ${nodeName}, status = $status)"
    else
		echo "Process $pid fail    (node: ${nodeName}, status = $status)"
		total_status=1
    fi

done

echo "----------------------------------------------------"

exit $total_status




















