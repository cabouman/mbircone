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



# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"


echo "--------------------------------------------------------------------------"
echo "Compiling plainParams Code ------------------------------------------------"
echo "--------------------------------------------------------------------------"
cd ../code/plainParams/
make clean
make
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi
cd ~-

echo "--------------------------------------------------------------------------"
echo "Compiling Inversion Code -------------------------------------------------"
echo "--------------------------------------------------------------------------"
bash ../code/ConeBeam/Inversion/./make.sh all
if [[  $? != 0 ]]; then generalError "$0 $@"; exit 1; fi

