#!/bin/bash
# This script just installs mbircone along with requirements of mbircone, demos, and documation. 
# However, it does not remove the existing installation of mbircone.

cd ..
pip install -r requirements.txt
pip install -r demo/requirements_demo.txt
pip install -r docs/requirements.txt 
pip install .
cd dev_scripts

