# MBIR Cone

Python Package for Cone Beam Computed Tomography reconstruction. 

Full documentation is available at [mbircone_docs](https://mbircone.readthedocs.io).

To cite this software package, please use the bibtext entry at [cite_mbircone](https://mbircone.readthedocs.io/en/latest/credits.html).

**Warning:** This is a pre-release version of code that is still under development.

## Distribution Statement

Distribution Statement A. Approved for public release: distribution unlimited (88ABW-2020-0895).

If you publish results based on this code, please cite the following paper:
> Thilo Balke, Soumendu Majee, Gregery T. Buzzard, Scott Poveromo, Patrick Howard, Michael A. Groeber, John McClure, Charles A. Bouman "Separable Models for cone-beam MBIR Reconstruction," Proceedings of the IS&T International Symposium on Electronic Imaging, Computational Imaging XVI, pp. 181-1 to 181-7, 2018.

For other OpenMBIR packages see: https://github.com/cabouman/OpenMBIR-Index


## Clone Package

Clone the package by running the following commands:
```
git clone git@github.com:cabouman/mbircone.git
cd mbircone
```


## Installation Package

There are two options for installing mbircone. Both approaches will require that you first install the `gcc` compiler and the associated `omp` libraries as per instructions here: https://svmbir.readthedocs.io/en/latest/install.html

**Option 1: Easy Installation:**
The easiest way to install is to use the available bash scripts:
```
cd mbircone/dev_scripts
source ./clean_install_all.sh
```

The script will create and activate a conda envirnoment named ``mbircone``, install the package and all its requirements, and build the documentation.

**Option 2: Manual Installation:**

1) Create conda environment
```
conda create -n mbircone python=3.8
```
2) Activate conda environment
```
conda activate mbircone
```
3) Install requirements
```
pip install -r requirements.txt
```
4) Install package
```
pip install .
```

## Run demos
1) Install demo requirements
```
cd demo
pip install -r requirements_demo.txt
```
2) Run basic demo
```
python demo_3D_shepp_logan.py
```
3) Run MACE demo
```
python demo_mace3D.py
```
4) (Optional) Run NSI dataset demo. (Note: this is a long demo with an estimated run time of 30-60 minutes!)
```
python demo_nsi_preprocess.py
```

5) Result visualization: 

Please go to ```demo/output/``` to look at phantom, sinogram, and reconstruction images

6) In the case where exceptions occur when downloading data, please check your internet connection. If you replaced the default url with the url of your own dataset, please make sure that the url is correct, and points to a public webpage.


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
