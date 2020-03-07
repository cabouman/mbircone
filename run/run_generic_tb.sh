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

################################################
# Make
################################################
./makeall.sh &&

./multiNode_setup.sh init &&

################################################
################################################
# 4D

# ./4D_cloning.sh 1 &&

# preprocessing
../code/Recon_4D/Inversion/inversion_4D_multiNode_smart.sh ../control/Recon_4D/master.txt preprocessing && 

# genSysMatrix
../code/Recon_4D/Inversion/inversion_4D_multiNode_smart.sh ../control/Recon_4D/master.txt genSysMatrix && 

# initialize
../code/Recon_4D/Inversion/inversion_4D_multiNode_smart.sh ../control/Recon_4D/master.txt initialize && 

# Recon
../code/Recon_4D/Inversion/inversion_4D_multiNode_smart.sh ../control/Recon_4D/master.txt Recon && 

# runall
../code/Recon_4D/Inversion/inversion_4D_multiNode_smart.sh ../control/Recon_4D/master.txt runall && 

# runall_exceptRecon
../code/Recon_4D/Inversion/inversion_4D_multiNode_smart.sh ../control/Recon_4D/master.txt runall_exceptRecon && 

# multiResolution
../code/Recon_4D/Inversion/inversion_4D_multiNode_smart.sh ../control/Recon_4D/master.txt multiResolution && 

# changePreprocessing
../code/Recon_4D/Inversion/inversion_4D_multiNode_smart.sh ../control/Recon_4D/master.txt changePreprocessing && 

# # CADMM
./4D_CADMM.sh 2>&1 | tee logs/log_CADMM.log


################################################


:
