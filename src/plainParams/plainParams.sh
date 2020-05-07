#!/bin/sh -l


source $(dirname ${0})/subroutines_plainParams.sh


###


while getopts :a:m:F:f:v:rh option
do
	case $option in
		a) get_set="${OPTARG}";;
		m) masterFileName="${OPTARG}";;
		F) masterField="${OPTARG}";;
		f) subField="${OPTARG}";;
		v) value="${OPTARG}";;
		r) resolve="true";;
		h) usage; exit 1;;
		?)
			plainParamsError "$0 $@"
			>&2 echo "    Unknown option -${OPTARG}!"
			usage
			exit 1;;
	esac
done

# >&2 echo "--------------------------------------"
# >&2 echo "get_set = ${get_set}"
# >&2 echo "masterFileName = ${masterFileName}"
# >&2 echo "masterField = ${masterField}"
# >&2 echo "subField = ${subField}"
# >&2 echo "value = ${value}"
# >&2 echo "--------------------------------------"


########  Check if options complete/correct
if [[ -z ${get_set} ]] || [[ -z ${masterFileName} ]] || [[ -z ${masterField} ]] 
then
	missingOption
	plainParamsError "$0 $@"
	exit 1
fi

if [[ ${get_set} == "set" ]]; then
	if [[ -z ${value} ]]; then
		missingOption
		plainParamsError "$0 $@"
		exit 1
	fi
	if [[ ${resolve} == "true" ]]; then
		>&2 echo "    Don't use option -r when writing to file"
		plainParamsError "$0 $@"
		exit 1
	fi
elif [[ ${get_set} == "get" ]]; then
	if [[ ! -z ${value} ]] ; then
		>&2 echo "    Don't use option -v when reading from file"
		plainParamsError "$0 $@"
		usage
		exit 1
	fi
fi

if [[ ${get_set} != "set" ]] && [[ ${get_set} != "get" ]]; then
	plainParamsError "$0 $@"
	>&2 echo '    Use "get" or "set" as option with -a'
	usage
	exit 1
fi

###############################################
### Main section

if [[ ! -z ${subField} ]] 
then  #### This is for accessing the subfiles
	if [[ ${get_set} == "get" ]]; then
		value=$(getSubFileField "${masterFileName}" "${masterField}" "${subField}")
		if [[  $? != 0 ]]; then plainParamsError "$0 $@"; exit 1; fi

		if [[ ${resolve} == "true" ]]; then
			subsubFile_rel="${value}"

			subFile_rel=$(getFieldFromFile "${masterField}" "${masterFileName}")
			subFile_abs=$(absolutePath_realativeTo "${subFile_rel}" "${masterFileName}")
			subsubFile_abs=$(absolutePath_realativeTo "${subsubFile_rel}" "${subFile_abs}")
			value="${subsubFile_abs}"
		fi

		echo "${value}"

	elif [[ ${get_set} == "set" ]]; then
		(setSubFileField "${masterFileName}" "${masterField}" "${subField}" "${value}"   )
		if [[  $? != 0 ]]; then plainParamsError "$0 $@"; exit 1; fi
	fi


else  #### This is for accessing the master file
	if [[ ${get_set} == "get" ]]; then
		value=$(getFieldFromFile "${masterField}" "${masterFileName}")
		if [[  $? != 0 ]]; then plainParamsError "$0 $@"; exit 1; fi

		if [[ ${resolve} == "true" ]]; then
			subFile_rel="${value}"
			
			subFile_abs=$(absolutePath_realativeTo "${subFile_rel}" "${masterFileName}")
			value="${subFile_abs}"
		fi

		echo "${value}"

	elif [[ ${get_set} == "set" ]]; then
		(setFieldInFile "${masterField}" "${value}" "${masterFileName}")
		if [[  $? != 0 ]]; then plainParamsError "$0 $@"; exit 1; fi
	fi
fi







