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

usage()
{
    >&2 echo "Usage:"
    >&2 echo "argument:    help"    
    >&2 echo "             all"
    >&2 echo "             clean"  

}



# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

if [[ $# -ne 1 ]]; then
    messageError "Number of arguments incorrect"
    generalError "$0 $@"
    usage
    exit 1
fi

mode="${1}"


SOURCES="allocate.c MBIRModularUtilities3D.c io3d.c computeSysMatrix.c icd3d.c recon3DCone.c main.c plainParams.c"
OBJECTS="allocate.o MBIRModularUtilities3D.o io3d.o computeSysMatrix.o icd3d.o recon3DCone.o main.o plainParams.o"
EXECUTABLE="main"



if [[ "${mode}" = 'help' ]]; then

    echo help me
    usage

elif [[ "${mode}" = "all" ]]; then

    echo make all the things
    echo "${scriptDir}"
    cd "${scriptDir}"

    set -x
        icc -fopenmp -O3 -Wall -pedantic -c ${SOURCES}
    { STATUS=$?; set +x; } 2>/dev/null
    if [[  $STATUS != 0 ]]; then generalError "$0 $@"; exit 1; fi

    set -x
        icc -fopenmp -O3 -Wall -pedantic ${OBJECTS}  -o ${EXECUTABLE}
    { STATUS=$?; set +x; } 2>/dev/null
    if [[  $STATUS != 0 ]]; then generalError "$0 $@"; exit 1; fi

    echo put all object and executables in the bin
    mv ${OBJECTS} ${EXECUTABLE} ../bin
    

# -------- Help --------------------------------------------

elif [[ "${mode}" = "clean" ]]; then

    echo clean all the things
    cd "${scriptDir}"
    rm ${OBJECTS} ${EXECUTABLE}
    cd ../bin
    rm ${OBJECTS} ${EXECUTABLE} 



# -------- Error --------------------------------------------
else

    messageError "Mode ${mode} is unknown"
    generalError "$0 $@"
    usage
    exit 1
fi

exit 0




