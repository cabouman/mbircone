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

################################################
# Make
################################################
./makeall.sh

################################################
################################################
# 3D

# ./Inversion_preprocessing.sh 1 2>&1 | tee log_preproc.log
# ./Inversion_runall.sh 1 2>&1 | tee log_Inv_runall.log
# ./Inversion_Recon.sh 1 2>&1 | tee log_Inv_Recon.log

# ./PlugAndPlay_run.sh 1 2>&1 | tee log_pnp.log

# Jig Correction

# ./FOV_correct.sh 1 correct_FOV 2>&1 | tee log_correction.log
# ./Inversion_Recon.sh 1 2>&1 | tee log_inv.log
# ./FOV_correct.sh 1 correct_FOV 2>&1 | tee log_correction.log
# ./Inversion_Recon.sh 1 2>&1 | tee log_inv.log



################################################
################################################
# 4D

# ./4D_cloning.sh 1
# ./4D_preprocessing.sh 2>&1 | tee log_preproc.log
# ./4D_inversion_multinode.sh preprocessing | tee log_preproc.log

./4D_inversion_multinode.sh genSysMatrix | tee log_inv.log
./4D_inversion_multinode.sh initialize | tee log_inv.log

# ./4D_inversion.sh genSysMatrix 2>&1 | tee log_inv.log
# ./4D_inversion.sh initAll_with_Recon 2>&1 | tee log_inv.log
# ./4D_inversion.sh initAll_with_FirstRecon 2>&1 | tee log_inv.log

# ./4D_inversion.sh reuseLastRecon 2>&1 | tee log_inv.log
# ./4D_inversion.sh Recon 2>&1 | tee log_inv.log
# ./4D_inversion_multinode.sh Recon 2>&1 | tee log_inv.log

# ./4D_inversion.sh runall 2>&1 | tee log_inv.log
# ./4D_inversion_multinode.sh runall | tee log_inv.log


./4D_consensus.sh 2>&1 | tee log_consensus.log



# Jig Correction

# ./4D_corrections.sh correct_FOV 2>&1 | tee log_correction.log
# ./4D_inversion.sh Recon 2>&1 | tee log_inv.log


################################################
# Use 3D recon for 4D jigcorrection
################################################
################################################
# ./4D_cloning.sh 1
# ./4D_inversion_multinode.sh preprocessing | tee log_preproc.log
# ./4D_inversion_multinode.sh genSysMatrix 2>&1 | tee log_inv.log
# ./4D_inversion.sh initAll_with_FirstRecon 2>&1 | tee log_inv.log

# ./4D_corrections.sh correct_FOV 2>&1 | tee log_correction.log
# ./4D_inversion_multinode.sh Recon | tee log_inv.log

