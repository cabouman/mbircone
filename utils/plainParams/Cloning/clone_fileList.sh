#!/usr/bin/env bash

# clone_fileList.sh -c <copyMode> -F <fileList_Fname> -p <prefix> -s <suffix>
# where
# <copyMode> = "shallow" or "deep" 
# <fileList_Fname> is mandatory. 
# both prefix and suffix cannot be empty
# works even if no extension. any relative or absolute path

generalError()
{
    >&2 echo "ERROR in $(basename ${0})"
    >&2 echo "When executing command \"bash $@\""
    >&2 echo
}

messageIn()
{
    >&2 echo "Error in ${1}"
}

while getopts ":c:F:f:p:s:" option
do
    case $option in
        c) copyMode="${OPTARG}";;
        F) fileList_Fname=$(readlink -f "${OPTARG}");;
        p) prefix="${OPTARG}";;
        s) suffix="${OPTARG}";;
        ?)
            >&2 echo "    Unknown option -${OPTARG}!"           
            exit 1;;
    esac
done


cd $(dirname "$0")
BASEDIR=$(pwd)

plainParamsFile_rel="../plainParams.sh"
plainParamsFile=$(readlink -f "${plainParamsFile_rel}");

# check for faulty arguments
if [[ ${prefix} == "" ]] && [[ ${suffix} == "" ]]; then
    >&2 echo "    both prefix and suffix empty !"
      exit 1
fi

# echo "fileList_Fname = $fileList_Fname"
fileList=($(cat "${fileList_Fname}"))
numFiles=${fileList[0]}
# echo "numFiles = $numFiles"

fileListDir=$(dirname ${fileList_Fname})
if [[  $? != 0 ]]; then messageIn "clone_fileList"; generalError "$0 $@"; exit 1; fi

for (( index=1; index<=$numFiles; index++ ))
do

    filePath=${fileList[${index}]}
    if [[  $? != 0 ]]; then messageIn "reading file in file list"; generalError "$0 $@"; exit 1; fi

    filePath_name=${filePath##*/}
    filePath_dir=${filePath%${filePath_name}}
    filePath_name_woExt=${filePath_name%.*}
    filePath_Ext=${filePath_name##${filePath_name_woExt}}
    filePath_name_woExt_new=${prefix}${filePath_name_woExt}${suffix}
    filePath_new=${filePath_dir}${filePath_name_woExt_new}${filePath_Ext}
    if [[  $? != 0 ]]; then messageIn "cloning file in file list"; generalError "$0 $@"; exit 1; fi


    if [[ ${copyMode} == "shallow" ]]; then

        # echo "shallow"
        fileList[${index}]=${filePath_new}
        if [[  $? != 0 ]]; then messageIn "setting file in file list"; generalError "$0 $@"; exit 1; fi

    elif [[ ${copyMode} == "deep" ]]; then

        # echo "deep"
        cd ${fileListDir}
        filePath_abs=$(readlink -f "${filePath}")
        filePath_new_abs=$(readlink -f "${filePath_new}")
        cd "$BASEDIR"
        if [[  $? != 0 ]]; then messageIn "getting absolute path"; generalError "$0 $@"; exit 1; fi

        cp "${filePath_abs}" "${filePath_new_abs}"
        if [[  $? != 0 ]]; then messageIn "copying"; generalError "$0 $@"; exit 1; fi

        fileList[${index}]=${filePath_new}
        if [[  $? != 0 ]]; then messageIn "setting file in file list"; generalError "$0 $@"; exit 1; fi

    else
      >&2 echo "    Unknown copyMode ${copyMode} !"
      exit 1
    fi

done

echo $numFiles > ${fileList_Fname}
for (( index=1; index<=$numFiles; index++ ))
do
    line=${fileList[${index}]}
    echo $line >> ${fileList_Fname}
done


