#!/usr/bin/env bash

# copy_fileList.sh -s <fileList_src_Fname> -d <fileList_dest_Fname> 


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

while getopts ":s:d:" option
do
    case $option in
        s) fileList_src_Fname=$(readlink -f "${OPTARG}");;
        d) fileList_dest_Fname=$(readlink -f "${OPTARG}");;
        ?)
            >&2 echo "    Unknown option -${OPTARG}!"           
            exit 1;;
    esac
done


cd $(dirname "$0")
BASEDIR=$(pwd)

plainParamsFile_rel="../plainParams.sh"
plainParamsFile=$(readlink -f "${plainParamsFile_rel}");


fileList_src=($(cat "${fileList_src_Fname}"))
numFiles_src=${fileList_src[0]}
if [[  $? != 0 ]]; then messageIn "reading source file list"; generalError "$0 $@"; exit 1; fi

fileList_src_dir=$(dirname ${fileList_src_Fname})
if [[  $? != 0 ]]; then messageIn "clone_fileList"; generalError "$0 $@"; exit 1; fi

echo "${numFiles_src}" > ${fileList_dest_Fname}

for (( index=1; index<=$numFiles_src; index++ ))
do
    filePath=${fileList_src[${index}]}
    if [[  $? != 0 ]]; then messageIn "reading file in file list"; generalError "$0 $@"; exit 1; fi

    cd ${fileList_src_dir}
    filePath_abs=$(readlink -f "${filePath}")
    cd "$BASEDIR"
    if [[  $? != 0 ]]; then messageIn "getting absolute path"; generalError "$0 $@"; exit 1; fi

    echo $filePath_abs >> ${fileList_dest_Fname}

done


