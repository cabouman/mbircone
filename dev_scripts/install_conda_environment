#!/bin/bash
# This script destroys the conda environment named "mbircone" and recreates it.

# Create and activate new conda environment
cd ..
conda deactivate
conda remove env --name mbircone --all
conda create --name mbircone python=3.8
conda activate mbircone
cd dev_scripts

