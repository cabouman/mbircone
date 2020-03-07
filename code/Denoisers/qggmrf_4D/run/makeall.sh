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

./cleanall.sh
if [[  $? != 0 ]]; then messageError "executing \"./cleanall.sh\"" ; generalError "$0 $@"; exit 1; fi


echo ""
echo "###########################################################"
echo "                Compiling Code"
echo "###########################################################"

echo ""
echo "************    Compiling Plain Params          ************"
cd ../../../plainParams/
if [[  $? != 0 ]]; then messageError "executing \"cd ../../../plainParams\"" ; generalError "$0 $@"; exit 1; fi
make
if [[  $? != 0 ]]; then messageError "executing make in plainParams/" ; generalError "$0 $@"; exit 1; fi
cd "${scriptDir}"

echo ""
echo "************    Compiling Libraries          ************"
cd ../C_Code/CLibraries/
if [[  $? != 0 ]]; then messageError "executing \"cd CLibraries/\"" ; generalError "$0 $@"; exit 1; fi
make
if [[  $? != 0 ]]; then messageError "executing make in CLibraries/" ; generalError "$0 $@"; exit 1; fi
cd "${scriptDir}"

echo ""
echo "************    Compiling Recon Code     ************"
cd ../C_Code/Recon/
if [[  $? != 0 ]]; then messageError "executing \"cd Recon/\"" ; generalError "$0 $@"; exit 1; fi
make
if [[  $? != 0 ]]; then messageError "executing make in Recon/" ; generalError "$0 $@"; exit 1; fi
cd "${scriptDir}"
