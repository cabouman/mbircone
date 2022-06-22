#!/bin/bash
# This script destroys the conda environment named "mbircone" and recreates it.

# Create and activate new conda environment
cd ..
conda deactivate
conda remove env --name mbircone_dev --all
conda create --name mbircone_dev python=3.8
conda activate mbircone_dev
cd dev_scripts

