#!/bin/bash
# This script destroys the conda environment named "mbircone" and uninstalls mbircone.
# It then creates an "mbircone" environment and reinstalls mbircone along with the documentation and demo requirements.

# Clean out old installation
yes | source clean_mbircone.sh

# Destroy conda environement named mbircone and reinstall it
yes | source reinstall_conda_environment.sh
conda activate mbircone

# Install mbircone
yes | source install_mbircone.sh

# Build documentation
yes | source install_docs.sh

