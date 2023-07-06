#!/bin/bash
# This script destroys the conda environment named "mbircone" and recreates it.

# Create and activate new conda environment
# First check if the target environment is active and deactivate if so
NAME=mbircone

ENV_STRING=$((conda env list) | grep $NAME)
if [[ $ENV_STRING == *$NAME* ]]; then
    conda deactivate
fi
cd ..

# Create and activate new conda environment
conda remove env --name mbircone --all
conda create --name mbircone python=3.8
conda activate mbircone

cd dev_scripts

