# MBIR Cone

Warnning: Prerelease version - This code is under development
Python Package for Cone Beam reconstruction.
Based on the OpenMBIR-ConeBeam c package.

Full documentation is available at: https://mbircone.readthedocs.io



## Distribution Statement

Distribution Statement A. Approved for public release: distribution unlimited (88ABW-2020-0895).
If you publish results based on this code, please cite the following paper:
> Thilo Balke, Soumendu Majee, Gregery T. Buzzard, Scott Poveromo, Patrick Howard, Michael A. Groeber, John McClure, Charles A. Bouman "Separable Models for cone-beam MBIR Reconstruction," proceedings of the IS&T International Symposium on Electronic Imaging, Computational Imaging XVI, pp. 181-1 to 181-7, 2018.

For other OpenMBIR packages see: https://github.com/cabouman/OpenMBIR-Index

## Installation
In order to install the package, clone the repository to your computer, and run the bash script named ``run_clean_install``.
The script will create and activate a conda envirnoment named ``mbircone``, install the package and all its requirements, and build the documentation.

The information below provides additional detail.

1) Clone Repository and enter
```
git clone https://github.com/cabouman/mbircone.git
cd mbircone
```

2) Create conda environment
```
conda create -n mbircone python=3.8
```
3) Activate conda environment
```
conda activate mbircone
```
4) Install requirements
```
pip install -r requirements.txt
```
5) Install package
```
pip install .
```

## Run demos
1) Install demo requirements
```
cd demo
pip install -r requirements.txt
```
2) Run basic demo
```
python demo_3D_shepp_logan.py
```
3) Run demo
```
python demo_mace3D.py
```

3) Result visualization: 

Please go to ```demo/output/mace3D/``` to look at phantom, sinogram, and reconstruction images

4) In the case where exceptions occur when downloading data, please check your internet connection. If you replaced the default url with the url of your own dataset, please make sure that the url is correct, and points to a public webpage.


## Build documentation in local folder
1) Install docs requirements
```
cd docs
pip install -r requirements.txt
```
2) Build documentation
```
MBIRCONE_BUILD_DOCS=true make html
```
3) Open documentation
```
cd build/html
open(double clicks) index.html
```
