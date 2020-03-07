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
cd ../../plainParams/
if [[  $? != 0 ]]; then messageError "executing \"cd ../../plainParams\"" ; generalError "$0 $@"; exit 1; fi
make
if [[  $? != 0 ]]; then messageError "executing make in plainParams/" ; generalError "$0 $@"; exit 1; fi
cd "${scriptDir}"

echo ""
echo "************    Compiling Libraries          ************"
cd ../Code/0A_CLibraries/
if [[  $? != 0 ]]; then messageError "executing \"cd 0A_CLibraries/\"" ; generalError "$0 $@"; exit 1; fi
make
if [[  $? != 0 ]]; then messageError "executing make in 0A_CLibraries/" ; generalError "$0 $@"; exit 1; fi
cd "${scriptDir}"

echo ""
echo "************    Compiling System Matrix Code ************"
cd ../Code/0B_GenSysMatrix/
if [[  $? != 0 ]]; then messageError "executing \"cd 0B_GenSysMatrix/\"" ; generalError "$0 $@"; exit 1; fi
make
if [[  $? != 0 ]]; then messageError "executing make in 0B_GenSysMatrix/" ; generalError "$0 $@"; exit 1; fi
cd "${scriptDir}"

echo ""
echo "************    Compiling Initialization Code   *********"
cd ../Code/0C_Initialize/
if [[  $? != 0 ]]; then messageError "executing \"cd 0C_Initialize/\"" ; generalError "$0 $@"; exit 1; fi
make
if [[  $? != 0 ]]; then messageError "executing make in 0C_Initialize/" ; generalError "$0 $@"; exit 1; fi
cd "${scriptDir}"

echo ""
echo "************    Compiling Inversion Code     ************"
cd ../Code/0D_Inversion/
if [[  $? != 0 ]]; then messageError "executing \"cd 0D_Inversion/\"" ; generalError "$0 $@"; exit 1; fi
make
if [[  $? != 0 ]]; then messageError "executing make in 0D_Inversion/" ; generalError "$0 $@"; exit 1; fi
cd "${scriptDir}"
