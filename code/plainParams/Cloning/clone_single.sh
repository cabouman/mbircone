#!/usr/bin/env bash

# clone_single.sh -c <copyMode> -P <plainParams.sh> -M <masterFile> -F <masterField> -f <subField> -p <prefix> -s <suffix>
# where
# <copyMode> = "shallow" or "deep" 
# <masterFile> is mandatory. Just this field relates to the master file itself
# <masterField> is optional.
# <subField> is optional (when masterField is nonempty)
# both prefix and suffix cannot be empty
# works even if no extension. any relative or absolute path

while getopts ":c:M:F:f:p:s:" option
do
    case $option in
        c) copyMode="${OPTARG}";;
        M) master=$(readlink -f "${OPTARG}");;
        F) masterField="${OPTARG}";;
        f) subField="${OPTARG}";;
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

# Find master file details
master_name=${master##*/}
master_dir=${master%${master_name}}

# clone
if [  -z ${masterField} ]; then
    # clone master
    master_name_woExt=${master_name%.*}
    master_Ext=${master_name##${master_name_woExt}}
    master_name_woExt_new=${prefix}${master_name_woExt}${suffix}
    master_new=${master_dir}${master_name_woExt_new}${master_Ext}

    if [[ ${copyMode} == "shallow" ]]; then

        # echo "shallow"
        echo "${master_new}"

    elif [[ ${copyMode} == "deep" ]]; then

        # echo "deep"
        cp "${master}" "${master_new}"
        echo "${master_new}"    

    else
      >&2 echo "    Unknown copyMode ${copyMode} !"
      exit 1
    fi

elif [  -z ${subField} ]; then
    # clone master field
    fieldPath=$(bash $plainParamsFile -a get -m "$master" -F "$masterField" )
    fieldPath_name=${fieldPath##*/}
    fieldPath_dir=${fieldPath%${fieldPath_name}}
    fieldPath_name_woExt=${fieldPath_name%.*}
    fieldPath_Ext=${fieldPath_name##${fieldPath_name_woExt}}
    fieldPath_name_woExt_new=${prefix}${fieldPath_name_woExt}${suffix}
    fieldPath_new=${fieldPath_dir}${fieldPath_name_woExt_new}${fieldPath_Ext}
    echo "${fieldPath_new}"


    if [[ ${copyMode} == "shallow" ]]; then

        # echo "shallow"
        $(bash $plainParamsFile -a set -m "$master" -F "$masterField" -v "$fieldPath_new" )

    elif [[ ${copyMode} == "deep" ]]; then

        # echo "deep"
        cd "$master_dir"
        fieldPath_abs=$(readlink -f "${fieldPath}")
        fieldPath_new_abs=$(readlink -f "$fieldPath_new")
        cd "$BASEDIR"
        cp "${fieldPath_abs}" "${fieldPath_new_abs}"

        $(bash $plainParamsFile -a set -m "$master" -F "$masterField" -v "$fieldPath_new" )

    else
      >&2 echo "    Unknown copyMode ${copyMode} !"
      exit 1
    fi

else
    # clone sub field
    fieldPath=$(bash $plainParamsFile -a get -m "$master" -F "$masterField" -f "$subField"  )
    fieldPath_name=${fieldPath##*/}
    fieldPath_dir=${fieldPath%${fieldPath_name}}
    fieldPath_name_woExt=${fieldPath_name%.*}
    fieldPath_Ext=${fieldPath_name##${fieldPath_name_woExt}}
    fieldPath_name_woExt_new=${prefix}${fieldPath_name_woExt}${suffix}
    fieldPath_new=${fieldPath_dir}${fieldPath_name_woExt_new}${fieldPath_Ext}

    if [[ ${copyMode} == "shallow" ]]; then

        # echo "shallow"
        $(bash $plainParamsFile -a set -m "$master" -F "$masterField" -f "$subField" -v "$fieldPath_new" )

    elif [[ ${copyMode} == "deep" ]]; then

        # echo "deep"
        cd "$master_dir"
        fieldPath_abs=$(readlink -f "${fieldPath}")
        fieldPath_new_abs=$(readlink -f "$fieldPath_new")
        cd "$BASEDIR"
        cp "${fieldPath_abs}" "${fieldPath_new_abs}"

        $(bash $plainParamsFile -a set -m "$master" -F "$masterField" -v "$fieldPath_new" )

    else
      >&2 echo "    Unknown copyMode ${copyMode} !"
      exit 1
    fi

fi 



