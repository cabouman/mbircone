# MBIR Cone

Python Package for Cone Beam reconstruction.
Based on the OpenMBIR-ConeBeam c package.


## Distribution Statement

Distribution Statement A. Approved for public release: distribution unlimited (88ABW-2020-0895).
If you publish results based on this code, please cite the following paper:
> Thilo Balke, Soumendu Majee, Gregery T. Buzzard, Scott Poveromo, Patrick Howard, Michael A. Groeber, John McClure, Charles A. Bouman "Separable Models for cone-beam MBIR Reconstruction," proceedings of the IS&T International Symposium on Electronic Imaging, Computational Imaging XVI, pp. 181-1 to 181-7, 2018.

For other OpenMBIR packages see: https://github.com/cabouman/OpenMBIR-Index

## Installation
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

## Run demo
1) Install demo requirements
```
cd demo
pip install -r requirements_demo.txt
```
2) Run demo
```
python demo_3D_shepp_logan.py
```

## Run MACE demo
1) Install MACE demo requirements
```
cd demo
pip install -r requirements_demo_mace.txt
```
2) Run demo
```
python demo_mace3D.py
```

3) Result visualization: 

Please go to ```demo/output/mace3D/``` to look at phantom, sinogram, and reconstruction images

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
