#!/bin/bash
# This script installs the documentation.
# You can view documentation pages from mbircone/docs/build/index.html .

# Build documentation
cd ../docs
MBIRCONE_BUILD_DOCS=true make html
cd ../dev_scripts
