#!/bin/bash
# This script destroys the conda environment named "mbircone" and uninstalls MBIRCONE.
# It then creates an "mbircone" environment and reinstalls MBIRCONE along with the documentation and demo requirements.


# Clean out old installation
source clean.sh

# Create and activate new conda environment
cd ..
conda deactivate 
conda remove env --name mbircone --all 
conda create -n mbircone python=3.8 
conda activate mbircone 

# Install requirements and package
pip install -r requirements.txt 
pip install . 
pip install -r demo/requirements_demo.txt 
pip install -r docs/requirements.txt 

# Build documentation
cd docs
MBIRCONE_BUILD_DOCS=true make html
cd ../dev_scripts

