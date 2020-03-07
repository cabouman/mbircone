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

numCores=${1}
echo "Setting num cores: ${numCores}"

# cd to where the script is
executionDir=$(pwd)
scriptDir=$(readlink -f $(dirname $0))
cd "${scriptDir}"

########################################################################
##########  Change number of cores in different parts of control #######
########################################################################
plainParamsFile=$(readlink -f "../code/plainParams/plainParams.sh")

# change recon params
master_inversion=$(readlink -f "../control/Inversion/QGGMRF/master.txt")
bash $plainParamsFile -a set -m $master_inversion -F reconParams -f numThreads -v $numCores

# change chunked denoise
master_chunked=$(readlink -f "../control/ChunkedDenoise/master.txt")
bash $plainParamsFile -a set -m $master_chunked -F maxNumProcesses -v $numCores