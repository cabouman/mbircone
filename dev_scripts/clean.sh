#!/bin/bash
# This script purges MBIRCONE

cd ..
/bin/rm mbircone/interface_cy_c.c
/bin/rm mbircone/*.so
/bin/rm -r build
/bin/rm -r dist
/bin/rm -r mbircone.egg-info

pip uninstall mbircone
cd dev_scripts

