#!/usr/bin/env bash

# To submit a job for SLURM clusters run: sbatch jobSub/SLURM_CB.sub 

# uncomment the following line to override mac's readlink with GNU readlink (more info in readme)
# alias readlink='greadlink'

./makeall.sh

master=$(readlink -f "../control/ConeBeam/master.txt")

bash ../src/ConeBeam/ConeBeam.sh "${master}" CBMODE_preprocessing

bash ../src/ConeBeam/ConeBeam.sh "${master}" CBMODE_INV_prepare

bash ../src/ConeBeam/ConeBeam.sh "${master}" CBMODE_INV_recon

