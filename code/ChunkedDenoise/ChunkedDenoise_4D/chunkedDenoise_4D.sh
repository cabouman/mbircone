#!/usr/bin/env bash

#########################################################################################
##### Display Functions

dispString()
{
	local name="${1}"
	local content="${2}"
	echo -e "${name} = ${content}"
}

dispBar()
{
	local c="${1}"
	echo "(1)$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c"
}

generalError()
{
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
}


usage()
{
	>&2 echo "Usage:"
	>&2 echo "-h                        : help"
	>&2 echo "-M <masterFile>   : "	
	>&2 echo "-m <denoisingMasterFile>  : "
	>&2 echo "-c <denoiserCloneFile>    : "
	>&2 echo "-d <denoiseScript>    : "
	>&2 echo "-L <denoiseScriptLang>    : "
	>&2 echo "-r <denoisingMultiRun>    : "
	>&2 echo "-p <prefix>               : "
	>&2 echo "-s <suffix>               : "
}

dispStart()
{
	echo "  /‾‾‾‾‾‾‾‾‾‾‾‾‾ ${1} ‾‾‾‾‾‾‾‾‾‾‾‾‾\\"
}

dispStop()
{
	echo "  \\_____________ ${1} _____________/"
	echo ""
}

messageIn()
{
    >&2 echo "Error in ${1}"
}

#########################################################################################
#### Global Functions



chunking_4D()
{
	cloned_chunkedDenoise_masterFiles=($(cat "${cloned_chunkedDenoise_masterFile_List}"))
	numTimePoints=${cloned_chunkedDenoise_masterFiles[0]}

	dispStop "Chunking 4D"

	################# Build command
	chunkCommand=""
	for (( t=0; t<$numTimePoints; t++ ))
	do
		index=$(($t + 1))
		cloned_chunkedDenoise_masterFile=${cloned_chunkedDenoise_masterFiles[${index}]}
		echo "Bulding chunking command(t = $t) on file $cloned_chunkedDenoise_masterFile"

		chunkCommand=$(echo -e "${chunkCommand}  ${matlab_chunkName}('${cloned_chunkedDenoise_masterFile}', '${plainParamsFile}'); disp(['t = ', num2str(${t}+1), ' of ', num2str(${numTimePoints})]);")

	done
	#echo "chunkCommand = ${chunkCommand}"

	################## Run chunking

	cd $matlab_chunkDir
	module load matlab
	# matlab -nodesktop -nosplash -r "${chunkCommand}" < /dev/null
	matlab -singleCompThread -nojvm -r "${chunkCommand}; exit"

	dispStop "Done Chunking"
}

get_ImageList()
{
	local __cloned_chunkedDenoise_masterFile=$1
	local __inputImageList_var=$2
	local __outputImageList_var=$3
	local __numProcesses_var=$4

	############### read master file
	local __inputImageListList=$(bash "${plainParamsFile}" -a get -m "${__cloned_chunkedDenoise_masterFile}" -F inputImageListList -r)
	if [[  $? != 0 ]]; then messageIn "get inputImageList"; generalError "$0 $@"; exit 1; fi
	local __outputImageListList=$(bash "${plainParamsFile}" -a get -m "${__cloned_chunkedDenoise_masterFile}" -F outputImageListList -r)
	if [[  $? != 0 ]]; then messageIn "get outputImageListList"; generalError "$0 $@"; exit 1; fi

	############### read file lists
	local __inputImageList=($(cat "${__inputImageListList}"))
	if [[  $? != 0 ]]; then messageIn "read inputImageList"; generalError "$0 $@"; exit 1; fi
	local __outputImageList=($(cat "${__outputImageListList}"))
	if [[  $? != 0 ]]; then messageIn "read outputImageList"; generalError "$0 $@"; exit 1; fi

	if [[ ${__inputImageList[0]} -eq ${__outputImageList[0]} ]]; then
		local __numProcesses=${__inputImageList[0]}
	else
		>&2 echo "Number of processes inconsistant"
		generalError "$0 $@"
		exit 1
	fi

	############### Return Values

	# assignment does not work
	# eval $__inputImageList_var=( "'$__inputImageList'" )
	# eval $__outputImageList_var="'$__outputImageList'"
	eval $__numProcesses_var="'$__numProcesses'"

	inputImageList=(${__inputImageList[*]})
	outputImageList=(${__outputImageList[*]})

}

populate_subLists()
{
	local filePath=$1
	local numFiles=$2

	local filePath_name=${filePath##*/}
    local filePath_dir=${filePath%${filePath_name}}
    local filePath_name_woExt=${filePath_name%.*}
    local filePath_Ext=${filePath_name##${filePath_name_woExt}}

    echo "${numFiles}" > ${filePath}

	for (( fileId = 1; fileId <= ${numFiles}; fileId++ )); do

		local suffix=_list_${fileId}
	    local filePath_name_woExt_new=${filePath_name_woExt}${suffix}
	    local filePath_new=${filePath_dir}${filePath_name_woExt_new}${filePath_Ext}

	    echo "${filePath_new}" >> ${filePath}

	done

}

gluing_4D()
{
	cloned_chunkedDenoise_masterFiles=($(cat "${cloned_chunkedDenoise_masterFile_List}"))
	numTimePoints=${cloned_chunkedDenoise_masterFiles[0]}

	dispStop "Gluing 4D"
	################# Build command
	glueCommand=""
	for (( t=0; t<$numTimePoints; t++ ))
	do
		index=$(($t + 1))
		cloned_chunkedDenoise_masterFile=${cloned_chunkedDenoise_masterFiles[${index}]}
		echo "Bulding gluing command(t = $t) on file $cloned_chunkedDenoise_masterFile"

		glueCommand=$(echo -e "${glueCommand}  ${matlab_glueName}('${cloned_chunkedDenoise_masterFile}', '${plainParamsFile}'); disp(['t = ', num2str(${t}+1), ' of ', num2str(${numTimePoints})]);")

	done
	#echo "glueCommand = ${glueCommand}"


	################## Run gluing

	cd $matlab_glueDir
	module load matlab
	# matlab -nodesktop -nosplash -r "${chunkCommand}" < /dev/null
	matlab -singleCompThread -nojvm -r "${glueCommand}; exit"

	dispStop "Done Gluing"

}

cleanFiles_4D()
{
	startend_string=$1

	cloned_chunkedDenoise_masterFiles=($(cat "${cloned_chunkedDenoise_masterFile_List}"))
	numTimePoints=${cloned_chunkedDenoise_masterFiles[0]}

	for (( t=0; t<$numTimePoints; t++ ))
	do
		
		index=$(($t + 1))

		cloned_chunkedDenoise_masterFile=${cloned_chunkedDenoise_masterFiles[${index}]}
		echo $cloned_chunkedDenoise_masterFile

		if [ "$startend_string" = "start" ] ; then
			isCleanStart=$(bash "${plainParamsFile}" -a get -m "${cloned_chunkedDenoise_masterFile}" -F isCleanStart )
			if [[  $isCleanStart == 1 ]]; then
				echo "Cleaning chunkedDenoise ${startend_string}: time ${t} of $numTimePoints"
				cleanFiles ${cloned_chunkedDenoise_masterFile} ${plainParamsFile}
			fi
		fi

		if [ "$startend_string" = "end" ] ; then
			isCleanEnd=$(bash "${plainParamsFile}" -a get -m "${cloned_chunkedDenoise_masterFile}" -F isCleanEnd )
			if [[  $isCleanEnd == 1 ]]; then
				echo "Cleaning chunkedDenoise ${startend_string}: time ${t} of $numTimePoints"
				cleanFiles ${cloned_chunkedDenoise_masterFile} ${plainParamsFile}
			fi
		fi

	done
}

#########################################################################################
#### 3D UTILITY Functions

cleanFiles()
{
	masterFile=$1
	plainParamsFile=$2

	# Clean Bianary files
	binaryFile=$(bash "${plainParamsFile}" -a get -m "${masterFile}" -F inputImageFName -r)
	folderSuffix=$(bash "${plainParamsFile}" -a get -m "${masterFile}" -F folderSuffix )
	binaryDir=$(dirname ${binaryFile})
	suffixedDir="${binaryDir}/*${folderSuffix}"
	echo "rm -rf ${suffixedDir}"
	rm -rf ${suffixedDir}

	# Clean inputImageListList txt files
	folderSuffix=$(bash "${plainParamsFile}" -a get -m "${masterFile}" -F folderSuffix )
	inputImageListList_file=$(bash "${plainParamsFile}" -a get -m "${masterFile}" -F inputImageListList -r)
	inputImageListList_dir=$(dirname ${inputImageListList_file})
	inputImageListList_base=$(basename ${inputImageListList_file})
	inputImageListList_nakedBase=${inputImageListList_base%.*}
	inputImageListList_chunkedDir=${inputImageListList_dir}/${inputImageListList_nakedBase}${folderSuffix}
	rm -rf ${inputImageListList_chunkedDir}

	# Clean outputImageListList txt files
	folderSuffix=$(bash "${plainParamsFile}" -a get -m "${masterFile}" -F folderSuffix )
	outputImageListList_file=$(bash "${plainParamsFile}" -a get -m "${masterFile}" -F outputImageListList -r)
	outputImageListList_dir=$(dirname ${outputImageListList_file})
	outputImageListList_base=$(basename ${outputImageListList_file})
	outputImageListList_nakedBase=${outputImageListList_base%.*}
	outputImageListList_chunkedDir=${outputImageListList_dir}/${outputImageListList_nakedBase}${folderSuffix}
	rm -rf ${outputImageListList_chunkedDir}
}





dispStart " * * * * * * * Start Chunked Denoising * * * * * * * "

while getopts :hM:m:c:r:d:p:s:L: option
do
	case $option in
		h)	usage; exit 1;;

		M) 	masterFile_4D="$(readlink -f ${OPTARG})"
			isMasterFile_4D=true;;
		m) 	denoisingMasterFile="$(readlink -f ${OPTARG})"
			isDenoisingMasterFile=true;;
		c) 	denoiserCloneFile="$(readlink -f ${OPTARG})"
			isDenoiserCloneFile=true;;
		d) 	denoiseScript="$(readlink -f ${OPTARG})"
			isDenoiseScript=true;;
		L)	denoiseScriptLang="${OPTARG}"
			isDenoiseScriptLang=true;;
		r) 	denoisingMultiRun="$(readlink -f ${OPTARG})"
			isDenoisingMultiRun=true;;
		p)	prefix="${OPTARG}"
			isPrefix=true;;
		s)	suffix="${OPTARG}"
			isSuffix=true;;

		?)
			>&2 echo "Unknown option -${option}!"
			generalError "$0 $@"
			exit 1;;
	esac
done


## includes ####
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"


plainParamsFile=$(readlink -f 			"../../plainParams/plainParams.sh")
multiClone4D_denoiserFile=$(readlink -f 	"multiClone_denoiser_4D.sh")
matlab_chunk=$(readlink -f ../chunk.m)
matlab_glue=$(readlink -f ../glue.m)
timeLog=$(readlink -f "elapsed_times.txt")

matlab_chunk_base=$(basename ${matlab_chunk})
matlab_chunkName=${matlab_chunk_base%.*}
matlab_chunkDir=$(dirname ${matlab_chunk})

matlab_glue_base=$(basename ${matlab_glue})
matlab_glueName=${matlab_glue_base%.*}
matlab_glueDir=$(dirname ${matlab_glue})

echo "Time summary: " > ${timeLog}


##### check if files exist
if [[ ! -e ${masterFile_4D} ]]; then echo "masterFile_4D does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${denoisingMasterFile} ]]; then echo "denoisingMasterFile does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${denoiserCloneFile} ]]; then echo "denoiserCloneFile does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${denoiseScript} ]]; then echo "denoiseScript does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${denoisingMultiRun} ]]; then echo "denoisingMultiRun does not exist!"; generalError "$0 $@"; exit 1; fi

if [[ ! -e ${masterFile_4D} ]]; then echo "masterFile_4D does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${plainParamsFile} ]]; then echo "plainParamsFile does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${matlab_chunk} ]]; then echo "matlab_chunk does not exist!"; generalError "$0 $@"; exit 1; fi
if [[ ! -e ${matlab_glue} ]]; then echo "matlab_glue does not exist!"; generalError "$0 $@"; exit 1; fi

if [[ ! "${isDenoiseScriptLang}" == "true" ]]; then echo "denoiseScriptLang not set!"; usage;  generalError "$0 $@" ; exit 1; fi

############### read 4D master file
noisyBinaryFName_timeList=$(bash $plainParamsFile -a get -m $masterFile_4D -F noisyBinaryFName_timeList -r)
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

denoisedBinaryFName_timeList=$(bash $plainParamsFile -a get -m $masterFile_4D -F denoisedBinaryFName_timeList -r)
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

orig_chunkedDenoise_masterFile=$(bash $plainParamsFile -a get -m $masterFile_4D -F orig_chunkedDenoise_masterFile -r)
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

chunkedDenoiseCloneFile=$(bash $plainParamsFile -a get -m $masterFile_4D -F chunkedDenoiseCloneFile -r)
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

cloned_chunkedDenoise_masterFile_List=$(bash $plainParamsFile -a get -m $masterFile_4D -F cloned_chunkedDenoise_masterFile_List -r)
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

timeSuffix=$(bash $plainParamsFile -a get -m $masterFile_4D -F timeSuffix)
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

commandList=$(bash $plainParamsFile -a get -m $masterFile_4D -F commandList -r)
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

multiNodeMaster=$(bash $plainParamsFile -a get -m $masterFile_4D -F multiNodeMaster -r)
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

multiNode_on=$(bash $plainParamsFile -a get -m $masterFile_4D -F multiNode_on)
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi


echo "noisyBinaryFName_timeList : $noisyBinaryFName_timeList"
echo "denoisedBinaryFName_timeList : $denoisedBinaryFName_timeList"
echo "orig_chunkedDenoise_masterFile : $orig_chunkedDenoise_masterFile"
echo "chunkedDenoiseCloneFile : $chunkedDenoiseCloneFile"
echo "cloned_chunkedDenoise_masterFile_List : $cloned_chunkedDenoise_masterFile_List"
echo "timeSuffix : $timeSuffix"
echo "commandList : $commandList"
echo "multiNodeMaster : $multiNodeMaster"
echo "multiNode_on : $multiNode_on"


if [[ $multiNode_on != 0 ]]; then
	echo "Multinode"
	hostNameList="$(bash "${plainParamsFile}" -a get -m "${multiNodeMaster}" -F hostNameList -r)"
	oldIFS="$IFS"; IFS=$'\n'; hostNameList_arr=($(<${hostNameList})); IFS="$oldIFS"
	numHostNames=${#hostNameList_arr[@]}
	echo "numHostNames: $numHostNames"
	$(bash $plainParamsFile -a set -m $orig_chunkedDenoise_masterFile -F maxNumProcesses -v $numHostNames )
	# set numAgents to numcores
	numCoresPerNode="$(bash "${plainParamsFile}" -a get -m "${multiNodeMaster}" -F numCoresPerNode)"
	$(bash $plainParamsFile -a set -m $multiNodeMaster -F numAgentsPerNode -v $numCoresPerNode )
fi

############### get noisy denoised and time points
noisyBinaryFName_times=($(cat "${noisyBinaryFName_timeList}"))
denoisedBinaryFName_times=($(cat "${denoisedBinaryFName_timeList}"))
if [[ ${noisyBinaryFName_times[0]} -eq ${denoisedBinaryFName_times[0]} ]]; then
	numTimePoints=${noisyBinaryFName_times[0]}
else
	>&2 echo "Number of time points inconsistant"
	generalError "$0 $@"
	exit 1
fi


############### clone ChunkedDenoise
echo "${numTimePoints}" > ${cloned_chunkedDenoise_masterFile_List}
for (( t=0; t<$numTimePoints; t++ ))
do

	# clone
	timeCloneSuffix=${timeSuffix}"_${t}"
	timeClonePrefix=""
	masterFile_temp=$(bash ${chunkedDenoiseCloneFile} -M "${orig_chunkedDenoise_masterFile}" -p "${timeClonePrefix}" -s "${timeCloneSuffix}"  )
	# echo $masterFile_temp
	if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi
	echo "${masterFile_temp}" >> ${cloned_chunkedDenoise_masterFile_List}

	index=$(($t + 1))

	# change inputImageFName
	noisyBinaryFName=${noisyBinaryFName_times[${index}]}
	$(bash $plainParamsFile -a set -m $masterFile_temp -F inputImageFName -v $noisyBinaryFName )

	# change outputImageFName
	denoisedBinaryFName=${denoisedBinaryFName_times[${index}]}
	$(bash $plainParamsFile -a set -m $masterFile_temp -F outputImageFName -v $denoisedBinaryFName )

done

############## cleaning 4D start
cleanFiles_4D "start"


############## chunking 4D
start_timestamp=$(date +%s)
chunking_4D
end_timestamp=$(date +%s)
elapsed_time=$(expr $end_timestamp - $start_timestamp)
echo "Chunking time: $elapsed_time" >> ${timeLog}



############### cloning 4D
cloned_chunkedDenoise_masterFiles=($(cat "${cloned_chunkedDenoise_masterFile_List}"))
numTimePoints=${cloned_chunkedDenoise_masterFiles[0]}
start_timestamp=$(date +%s)

dispStop "Cloning 4D "
for (( t=0; t<$numTimePoints; t++ ))
do
	index=$(($t + 1))

	cloned_chunkedDenoise_masterFile=${cloned_chunkedDenoise_masterFiles[${index}]}
	get_ImageList ${cloned_chunkedDenoise_masterFile} inputImageList outputImageList numProcesses

	if [[  $t == 0 ]]; then
		# init denoising master

		denoiser_masterListList=$(bash $plainParamsFile -a get -m $masterFile_4D -F denoiser_masterListList -r)
		if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

		populate_subLists ${denoiser_masterListList} ${numProcesses}
		echo "populated denoiser_masterListList: ${denoiser_masterListList}"
		denoiser_masterLists=($(cat "${denoiser_masterListList}"))
		
	fi

	for (( pid = 1; pid <= ${numProcesses}; pid++ )); do

		echo "   Cloning for process ${pid} time ${t}"
		fullSuffix="${suffix}_pid${pid}"

		if [[  $t == 0 ]]; then
			# init denoising master

			bash ${multiClone4D_denoiserFile} -m ${denoisingMasterFile} -c ${denoiserCloneFile} -i ${inputImageList[${pid}]} -o ${outputImageList[${pid}]} -p ${prefix} -s ${fullSuffix} -L ${denoiser_masterLists[$pid]} &
			if [[  $? != 0 ]]; then messageIn "cloning (pid $pid)"; generalError "$0 $@"; exit 1; fi
			
		else
			# update denoising master

			bash ${multiClone4D_denoiserFile} -m ${denoisingMasterFile} -c ${denoiserCloneFile} -i ${inputImageList[${pid}]} -o ${outputImageList[${pid}]} -p ${prefix} -s ${fullSuffix} -L ${denoiser_masterLists[$pid]} -u &
			if [[  $? != 0 ]]; then messageIn "cloning (pid $pid)"; generalError "$0 $@"; exit 1; fi
		fi

	done
	wait
	
done

dispStop "Done Cloning"

end_timestamp=$(date +%s)
elapsed_time=$(expr $end_timestamp - $start_timestamp)
echo "Cloning time: $elapsed_time" >> ${timeLog}




############## denoise 4D
> ${commandList}
start_timestamp=$(date +%s)
dispStart "Start Denoising"
for (( pid = 1; pid <= ${numProcesses}; pid++ )); do

	denoiserMasterFiles=($(cat "${denoiser_masterLists[${pid}]}"))
	numFiles="${denoiserMasterFiles[*]:0:1}"
	echo " Building denoising command: pid ${pid} numfiles: ${numFiles}"

	denoisermasterList_fName=${denoiser_masterLists[${pid}]}
	#echo "$denoisermasterList_fName"
	echo "${denoisingMultiRun} -n ${numFiles} -m ${denoisermasterList_fName} -d ${denoiseScript} -L ${denoiseScriptLang}" >> ${commandList}

done

oldIFS="$IFS"; IFS=$'\n'; temp_commandList_arr=($(cat ${commandList})); IFS="$oldIFS"
if [[ $multiNode_on != 0 ]]; then
	echo "Multinode"

	multiNode_commandList="$(bash "${plainParamsFile}" -a get -m "${multiNodeMaster}" -F commandList -r)"
	if [[  $? != 0 ]]; then messageError "resolving commandList" ; generalError "$0 $@"; exit 1; fi

	> ${multiNode_commandList}
	for temp_command in "${temp_commandList_arr[@]}"; do
		echo "${temp_command}" >> ${multiNode_commandList}
	done

	cd ${scriptDir}
	../../multiNode/./multiNode.sh "${multiNodeMaster}"
	
else
	echo "Singlenode"

	for temp_command in "${temp_commandList_arr[@]}"; do
		bash ${temp_command} &
	done
	wait
fi


dispStop "Done Denoising"
end_timestamp=$(date +%s)
elapsed_time=$(expr $end_timestamp - $start_timestamp)
echo "Denoising time: $elapsed_time" >> ${timeLog}




############## gluing 4D
start_timestamp=$(date +%s)
gluing_4D
end_timestamp=$(date +%s)
elapsed_time=$(expr $end_timestamp - $start_timestamp)
echo "Gluing time: $elapsed_time" >> ${timeLog}

############### cleaning 4D end
cleanFiles_4D "end"

dispStop " * * * * * * * Stop Chunked Denoising * * * * * * * "

cat ${timeLog}








