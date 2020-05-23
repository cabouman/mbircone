#################################################################################
#    On file level 
#################################################################################
getLineNumberOfFieldInFile(){
    local field=${1}
    local fileName=${2}
    local lineNumber
    lineNumber=$(awk "\$1==\"[${field}]\" {print NR+1}"    "${fileName}")
    if [[  $? != 0 ]]; then 
        (>&2 echo "    When trying to find \"${field}\" in file \"${fileName}\".")
        exit 1
    fi


    local numOccurences=$(echo $lineNumber | wc -w)
    local numLines=$(( $(wc -l < ${fileName}) + 1 ))

    # Test for uniqueness
    if [[ ${numOccurences} == 0 ]]; then
        (>&2 echo "    Field \"${field}\" not found in file \"${fileName}\".")
        exit 1
    fi
    if [[ ${numOccurences} -gt 1 ]]; then
        (>&2 echo "    Field \"${field}\" found ${numOccurences} times in file \"${fileName}\".")
        exit 1
    fi

    # Test whether there is a line after the keyword
    if [[ ${lineNumber} -gt ${numLines} ]]; then
        (>&2 echo "    Field \"${field}\" found in file \"${fileName} but file terminated early\".")
        exit 1
    fi

    # when everything is okay output line number
    echo "${lineNumber}"

}

getLineFromFile(){
    local lineNumber=${1}
    local fileName=${2}
    local numLines=$(( $(wc -l < ${fileName}) + 1 ))

    # Check if file is long enough
    if [[ ${lineNumber} -gt ${numLines} ]]; then
        (>&2 echo "    Tryied to read \"${lineNumber}\"-th line of file \"${fileName}\" which has ${numLines} lines \".")
        exit 1
    fi

    local value=$(awk "{if(NR==${lineNumber}) {print \$0}}"    "${fileName}")
    echo "${value}"
}

setLine2ValueInFile(){
    local lineNumber=${1}
    local value=${2}
    local fileName=${3}
    local numLines=$(( $(wc -l < "${fileName}") + 1 ))

    # Check if file is too short
    if [[ ${lineNumber} -gt ${numLines} ]]; then
        (>&2 echo "    Tryied to write \"${lineNumber}\"-th line of file \"${fileName}\" which has ${numLines} lines \".")
        exit 1
    fi

    local fileContent=$(awk "{  \
    if (NR == ${lineNumber}-1)    \
        {print \$0; getline; print \"${value}\"; }  \
    else                        \
        { print \$0;}           \
    }"                          \
    ${fileName})
    tempFile=$(mktemp)
    echo "${fileContent}" > ${tempFile}
    mv ${tempFile} ${fileName}
}

#################################################################################
#       On field level
#################################################################################
getFieldFromFile(){
    local field=${1}
    local fileName=${2}
    local lineNumber

    lineNumber=$(getLineNumberOfFieldInFile "${field}" "${fileName}")
    if [[  $? != 0 ]]; then 
        exit 1
    fi

    local line
    line=$(getLineFromFile ${lineNumber} "${fileName}")
    if [[  $? != 0 ]]; then 
        exit 1
    fi
    echo $line
}

setFieldInFile(){
    local field=${1}
    local value=${2}
    local fileName=${3}

    local lineNumber
    lineNumber=$(getLineNumberOfFieldInFile "${field}" ${fileName})
    if [[  $? != 0 ]]; then 
        exit 1
    fi

    setLine2ValueInFile ${lineNumber} "${value}" "${fileName}"
    if [[  $? != 0 ]]; then 
        exit 1
    fi
}

#################################################################################
#       On Sub file level
#################################################################################
getSubFileName(){
    local masterFileName=${1}
    local masterField=${2}

    local subFile_relPath
    subFile_relPath=$(getFieldFromFile "${masterField}" "${masterFileName}")
    if [[  $? != 0 ]]; then 
        exit 1
    fi


    local dir_backup=$(pwd)
    cd $(dirname ${masterFileName})
    local subFile_absPath
    subFile_absPath=$(readlink -f ${subFile_relPath})
    if [[  $? != 0 ]]; then
        (>&2 echo "    When resolving file name \"${subFile_relPath}\" from directory \"$(pwd)\"")
        exit 1
    fi
    cd ${dir_backup}

    echo $subFile_absPath
}
getSubFileField(){
    local masterFileName=${1}
    local masterField=${2}
    local subField=${3}

    local subFile_absPath
    subFile_absPath=$(getSubFileName "${masterFileName}" "${masterField}" )
    if [[  $? != 0 ]]; then 
        exit 1
    fi


    local value
    value=$(getFieldFromFile "${subField}" "${subFile_absPath}")
    if [[  $? != 0 ]]; then 
        exit 1
    fi

    echo ${value}

}

setSubFileField(){
    local masterFileName=${1}
    local masterField=${2}
    local subField=${3}
    local value=${4}

    local subFile_absPath
    subFile_absPath=$(getSubFileName "${masterFileName}" "${masterField}" )
    if [[  $? != 0 ]]; then 
        exit 1
    fi

    setFieldInFile "${subField}" "${value}" "${subFile_absPath}"
    if [[  $? != 0 ]]; then 
        exit 1
    fi
}


#################################################################################
#       Usage auxiliary
#################################################################################


missingOption(){
    >&2 echo "    Missing option"
    usage
}


usage(){
    >&2 echo '....................................................'
    >&2 echo '| Usage:                                           |'
    >&2 echo '|   -a <"get"/"set">          Reading or writing   |'
    >&2 echo '|   -m <masterFileName>       Path to master file  |'
    >&2 echo '|   -F <masterField>          Field of master file |'
    >&2 echo '|   -f <subField>         (*) Field of sub file    |'
    >&2 echo '|   -v <value>           (**) Value to write       |'
    >&2 echo '....................................................'
    >&2 echo '|   -r                  (***) Resolve path         |'
    >&2 echo '|   -h                        Usage information    |'
    >&2 echo '....................................................'
    >&2 echo '|   (*)   Omit for field in master file            |'
    >&2 echo '|   (**)  Omit for "get" option.                   |'
    >&2 echo '|   (***) Omit for "set" option                    |'
    >&2 echo '....................................................'

}

plainParamsError()
{
    >&2 echo "ERROR in plainParams.sh"
    >&2 echo "When executing command \"bash $@\""
}

#################################################################################
#
#################################################################################
absolutePath_realativeTo()
{
    rel="${1}"
    ref="${2}"
    local dir_backup=$(pwd)
    if [[ -d $ref ]]; then
        cd "${ref}"
    else
        cd "$(dirname ${ref})"
    fi
    local abs=$(readlink -f ${rel})
    cd ${dir_backup}
    echo ${abs}

}