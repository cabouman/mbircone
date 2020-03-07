#!/usr/bin/env bash
generalError()
{
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
}

dispString()
{
	local name="${1}"
	local content="${2}"
	>&2 echo "${name} = ${content}"
}

dispBar()
{
	local c="${1}"
	>&2 echo " (2)$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c$c"
}

usage()
{
	>&2 echo "Usage:"
	>&2 echo "-h                     : help"
	>&2 echo "-m <master>            : denoising master file"
	>&2 echo "-c <cloneFile>        : denoiser clone file"
	>&2 echo "-i <inputImageList>   : file with list of noisy image paths"
	>&2 echo "-o <outputImageList>  : file with list of denoised image paths"
	>&2 echo "-p <prefix>            : file name prefix"
	>&2 echo "-s <suffix>            : file name suffix"
	>&2 echo "-u <isUpdateMode>        : is update mode"
}

isUpdateMode=false

while getopts :hm:c:i:o:p:s:L:u option
do
	case $option in
		h)	usage; exit 1;;
		m) 	masterFile="$(readlink -f ${OPTARG})"
			isMasterFile=true;;
		c) 	cloneFile="$(readlink -f ${OPTARG})"
			isCloneFile=true;;
		i) 	inputImageList="$(readlink -f ${OPTARG})"
			isInputImageList=true;;
		o) 	outputImageList="$(readlink -f ${OPTARG})"
			isOutputImageList=true;;
		p)	prefix="${OPTARG}";;
		s)	suffix="${OPTARG}";;
		L)	denoiseMasterFileList="$(readlink -f ${OPTARG})"
			isdenoiseMasterFileList=true;;
		u)	isUpdateMode=true;;
		?)
			>&2 echo "Unknown option -${option}!"
			generalError "$0 $@"
			exit 1;;
	esac
done

# dispString masterFile ${masterFile}
# dispString inputImageList ${inputImageList}
# dispString outputImageList ${outputImageList}
# dispString denoiseMasterFileList ${denoiseMasterFileList}

# dispString isMasterFile ${isMasterFile}
# dispString isInputImageList ${isInputImageList}
# dispString isOutputImageList ${isOutputImageList}
# dispString isdenoiseMasterFileList ${isdenoiseMasterFileList}
# dispString isUpdateMode ${isUpdateMode}

##################### check args

if [[ ! "${isMasterFile}" == "true" ]]; then >&2 echo "masterFile not set!"; usage; generalError "$0 $@" ; exit 1; fi
if [[ ! "${isCloneFile}" == "true" ]]; then >&2 echo "cloneFile not set!"; usage; generalError "$0 $@" ; exit 1; fi
if [[ ! "${isInputImageList}" == "true" ]]; then >&2 echo "inputImageList not set!"; usage; generalError "$0 $@" ; exit 1; fi
if [[ ! "${isOutputImageList}" == "true" ]]; then >&2 echo "outputImageList not set!"; usage; generalError "$0 $@" ; exit 1; fi

if [[ ! -e ${masterFile} ]]; then >&2 echo "masterFile does not exist!"; usage; generalError "$0 $@" ; exit 1; fi
if [[ ! -e ${cloneFile} ]]; then >&2 echo "cloneFile does not exist!"; usage; generalError "$0 $@" ; exit 1; fi
if [[ ! -e ${inputImageList} ]]; then >&2 echo "inputImageList does not exist!"; usage; generalError "$0 $@" ; exit 1; fi
if [[ ! -e ${outputImageList} ]]; then >&2 echo "outputImageList does not exist!"; usage; generalError "$0 $@" ; exit 1; fi



######################## includes


executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))

cd "${scriptDir}"
plainParamsFile=$(readlink -f "../../plainParams/plainParams.sh")
if [[ ! -e ${plainParamsFile} ]]; then >&2 echo "plainParamsFile does not exist!"; generalError "$0 $@" ; exit 1; fi
cd "${executionDir}"



################## deal with file lists


inputImages=($(cat "${inputImageList}"))
outputImages=($(cat "${outputImageList}"))


if [[ ! "${inputImages[0]}" == $(( ${#inputImages[@]} - 1 )) ]] || [[ ! "${outputImages[0]}" == $(( ${#outputImages[@]} - 1 )) ]] || [[ ! "${numInputImages}" == "${numOutputImages}" ]] ; then
	>&2 echo "Number of images inconsistant"
	generalError "$0 $@"
	exit 1
else
	numImages="${inputImages[0]}"
	# >&2 echo "num images: $numImages"
fi

##################

if [[ ! "${isUpdateMode}" == "true" ]]; then
	# Fresh cloning

	echo "${numImages}" > ${denoiseMasterFileList}
else
	# Update mode

	denoiserMasterFiles=($(cat "${denoiseMasterFileList}"))
	numImages=${denoiserMasterFiles[0]}
fi

for (( i = 1; i <= ${numImages}; i++ )); do
	inputImage="${inputImages[${i}]}"
	outputImage="${outputImages[${i}]}"

	if [[ ! "${isUpdateMode}" == "true" ]]; then
		# Fresh cloning

		denoiseMasterFile=$(bash ${cloneFile} -M ${masterFile} -p "${prefix}" -s "${suffix}_${i}")
		if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

		noisyBinaryFName_timeList=$(bash ${plainParamsFile} -a get -m ${denoiseMasterFile} -F noisyBinaryFName_timeList -r)
		echo "1" > ${noisyBinaryFName_timeList}
		echo "${inputImage}" >> ${noisyBinaryFName_timeList}
		if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

		denoisedBinaryFName_timeList=$(bash ${plainParamsFile} -a get -m ${denoiseMasterFile} -F denoisedBinaryFName_timeList -r)
		echo "1" > ${denoisedBinaryFName_timeList}
		echo "${outputImage}" >> ${denoisedBinaryFName_timeList}
		if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

		echo "${denoiseMasterFile}" >> ${denoiseMasterFileList}
		if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

		echo "denoiseMasterFile: ${denoiseMasterFile}"

	else
		# Update mode

		denoiseMasterFile=${denoiserMasterFiles[$i]}
		if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

		echo "denoiseMasterFile: ${denoiseMasterFile}"

		noisyBinaryFName_timeList=$(bash ${plainParamsFile} -a get -m ${denoiseMasterFile} -F noisyBinaryFName_timeList -r)
		noisyBinaryFName_times=($(cat "${noisyBinaryFName_timeList}"))
		nT_old=${noisyBinaryFName_times[0]}
		nT=$(($nT_old + 1))
		#write new time
		sed -i "1s/.*/$nT/" ${noisyBinaryFName_timeList}
		echo "${inputImage}" >> ${noisyBinaryFName_timeList}
		if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

		denoisedBinaryFName_timeList=$(bash ${plainParamsFile} -a get -m ${denoiseMasterFile} -F denoisedBinaryFName_timeList -r)
		denoisedBinaryFName_times=($(cat "${denoisedBinaryFName_timeList}"))
		nT_old=${denoisedBinaryFName_times[0]}
		nT=$(($nT_old + 1))
		#write new time
		sed -i "1s/.*/$nT/" ${denoisedBinaryFName_timeList}
		echo "${outputImage}" >> ${denoisedBinaryFName_timeList}
		if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi
	fi

	
done



