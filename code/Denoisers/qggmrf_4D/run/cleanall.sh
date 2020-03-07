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

echo ""
echo "###########################################################"
echo "                Cleaning Code"
echo "###########################################################"

echo ""
echo "************    Cleaning Plain Params          ************"
cd ../../../plainParams/
if [[  $? != 0 ]]; then messageError "executing \"cd ../../../plainParams\"" ; generalError "$0 $@"; exit 1; fi
make clean
cd "${scriptDir}"

echo ""
echo "************    Cleaning Libraries          ************"
cd ../C_Code/CLibraries/
if [[  $? != 0 ]]; then messageError "executing \"cd CLibraries/\"" ; generalError "$0 $@"; exit 1; fi
make clean
cd "${scriptDir}"

echo ""
echo "************    Cleaning Recon Code     ************"
cd ../C_Code/Recon/
if [[  $? != 0 ]]; then messageError "executing \"cd Recon/\"" ; generalError "$0 $@"; exit 1; fi
make clean
cd "${scriptDir}"
