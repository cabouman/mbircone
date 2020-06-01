#!/usr/bin/env bash

# To submit a job for SLURM clusters run: sbatch jobSub/SLURM_CB.sub 

# uncomment the following line to override mac's readlink with GNU readlink (more info in readme)
# alias readlink='greadlink'

#./makeall.sh

cur=$(pwd)
cd ../
make all
cd ${cur}

master=$(readlink -f "../demo/params/master.txt")

bash ./ConeBeam.sh "${master}" CBMODE_preprocessing

bash ./ConeBeam.sh "${master}" CBMODE_INV_prepare

bash ./ConeBeam.sh "${master}" CBMODE_INV_recon

