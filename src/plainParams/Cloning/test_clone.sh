#!/usr/bin/env bash

cd $(dirname "$0")
BASEDIR=$(pwd)

master_source="Control_test/master.txt"
prefix="copy_"
suffix=""
# prefix="prefix"
# suffix="sufix"

./clone_inversion.sh -M "${master_source}" -p "${prefix}" -s "${suffix}" 


cd $BASEDIR