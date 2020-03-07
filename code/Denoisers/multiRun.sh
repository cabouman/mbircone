#!/usr/bin/env bash

usage()
{
	>&2 echo "Usage:"
	>&2 echo "-h                      : help"
	>&2 echo "-n <numFiles>           : number of master files"
	>&2 echo "-m <masterFiles>        : master files"
	>&2 echo "-d <denoiseScript>  : denoising matlab file"
}

generalError()
{
	>&2 echo "**********************************************";
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
    >&2 echo "**********************************************";
}

dispString()
{
	local name="${1}"
	local content="${2}"
	>&2 echo "${name} = ${content}"
}

while getopts :hn:m:d:L: option
do
	case $option in
		h)	usage; exit 1;;
		n) 	numFiles="${OPTARG}"
			isNumFiles=true;;
		m) 	masterList_fName="${OPTARG}"
			isMasterFiles=true;;
		d) 	denoiseScript="$(readlink -f ${OPTARG})"
			isDenoiseScript=true;;
		L)	denoiseScriptLang="${OPTARG}"
			isDenoiseScriptLang=true;;
	
		?)
			>&2 echo "Unknown option -${option}!"
			generalError "$0 $@"
			exit 1;;
	esac
done

##################### check args

if [[ ! "${isDenoiseScript}" == "true" ]]; then >&2 echo "denoise script not set!"; usage;  generalError "$0 $@" ; exit 1; fi
if [[ ! "${isMasterFiles}" == "true" ]]; then >&2 echo "master files fname not set!"; usage;  generalError "$0 $@" ; exit 1; fi
if [[ ! "${isNumFiles}" == "true" ]]; then >&2 echo "numFiles not set!"; usage;  generalError "$0 $@" ; exit 1; fi
if [[ ! "${isDenoiseScriptLang}" == "true" ]]; then >&2 echo "denoiseScriptLang not set!"; usage;  generalError "$0 $@" ; exit 1; fi

if [[ ! -e ${denoiseScript} ]]; then >&2 echo "denoiseScript does not exist!"; usage;  generalError "$0 $@" ; exit 1; fi





###################  resolve paths
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))

cd "${scriptDir}"
plainParamsFile=$(readlink -f "../plainParams/plainParams.sh")
if [[ ! -e ${plainParamsFile} ]]; then >&2 echo "plainParamsFile does not exist!"; usage;  generalError "$0 $@" ; exit 1; fi
cd "${executionDir}"

denoiseScript_base=$(basename ${denoiseScript})
denoiseScriptName=${denoiseScript_base%.*}
denoiseScriptDir=$(dirname ${denoiseScript})


################## Deal with arguments
oldIFS="$IFS"; IFS=$'\n'; masterFiles=($(cat ${masterList_fName})); IFS="$oldIFS"
unset masterFiles[0]
# dispString masterFiles[*] "${masterFiles[*]}"
if [[ ! "${numFiles}" == "${#masterFiles[@]}" ]]; then
	>&2 echo "Number of master files inconsistant"
	>&2 echo "numFiles = ${numFiles}"
	>&2 echo "#masterFiles[@] = ${#masterFiles[@]} from file: ${masterList_fName} "
	exit 1
fi

ranDenoisingCode=0
################## Switch denoise script type
if [ "${denoiseScriptLang}" == "MATLAB" ]; then

	################# Build command
	denoiseCommand=""
	for masterFile_rel in "${masterFiles[@]}"; do
		# dispString masterFiles[$i] ${masterFiles[$i]}
		#echo "master file: ${masterFile_rel}"
		masterFile=$(readlink -f ${masterFile_rel})
		if [[ ! -e ${masterFile} ]]; then >&2 echo "masterFile \"${masterFile}\" does not exist!";  generalError "$0 $@" ; exit 1; fi
		
		denoiseCommand=$(echo -e "${denoiseCommand}  ${denoiseScriptName}('${masterFile}', '${plainParamsFile}');")
		
	done
	#echo "denoiseCommand = ${denoiseCommand}"

	################## Run denoising

	cd $denoiseScriptDir
	module load matlab
	# matlab -nodesktop -nosplash -r "${denoiseCommand}" < /dev/null
	matlab -singleCompThread -nojvm -r "${denoiseCommand}; exit"

	ranDenoisingCode=1
fi

if [ "${denoiseScriptLang}" == "shell" ]; then	

	for masterFile_rel in "${masterFiles[@]}"; do
		# echo "master file: ${masterFile_rel}"
		masterFile=$(readlink -f ${masterFile_rel})
		if [[ ! -e ${masterFile} ]]; then >&2 echo "masterFile \"${masterFile}\" does not exist!";  generalError "$0 $@" ; exit 1; fi

		denoiseCommand="${denoiseScript} ${masterFile} ${plainParamsFile} "
		# echo "${denoiseCommand}"
		bash ${denoiseCommand}
		if [[  $? != 0 ]]; then >&2 echo "multiRun.sh: Error Denoising"; generalError "$0 $@"; exit 1; fi
	done

	ranDenoisingCode=1
fi

if [ ! $ranDenoisingCode == 1 ]; then >&2 echo "multiRun.sh: Denoising code did not run"; generalError "$0 $@"; exit 1; fi


