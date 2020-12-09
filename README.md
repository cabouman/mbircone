# MBIR Cone

Our goal is to write a Python Package for Cone Beam reconstruction.
This will be based on a Cython interface to the C-base OpenMBIR-ConeBeam package.


## Distribution Statement

Distribution Statement A. Approved for public release: distribution unlimited (88ABW-2020-0895).
If you publish results based on this code, please cite the following paper:
> Thilo Balke, Soumendu Majee, Gregery T. Buzzard, Scott Poveromo, Patrick Howard, Michael A. Groeber, John McClure, Charles A. Bouman "Separable Models for cone-beam MBIR Reconstruction," proceedings of the IS&T International Symposium on Electronic Imaging, Computational Imaging XVI, pp. 181-1 to 181-7, 2018.

For other OpenMBIR packages see: https://github.com/cabouman/OpenMBIR-Index

## Organization

The repository is organized as follows:
1) ```bin```: Contains binary executable and bash interface
2) ```demo```: Contains demo data, params and demo scripts
3) ```docs```: Contains documentation file
4) ```src```: Contains the c source code for conebeam reconstruction
5) ```utils```: Contains utility scripts for preprocessing, parameter manipulation, rendering and traditional FDK reconstruction

## Demo

The ```demo``` folder contains a preprocessing and a reconstruction demo (data, parameters, scripts etc).
These two demos can be run using the scripts ```demo_recon.sh``` and ```demo_preprocessing.sh```.
MATLAB is required for ```demo_preprocessing.sh``` but ```demo_recon.sh``` requires only a c compiler.
The two demos are independant, i.e. one can run ```demo_recon.sh``` without running ```demo_preprocessing.sh```.

### Reconstruction demo

This demo can be run using the script ```demo_recon.sh```.
Input data are in ```demo/inversion```, output data are also saved in ```demo/inversion```.

To submit a SLURM job in a HPC cluster with the demo, run ```sbatch jobSub/SLURM_CB.sub``` from the ```demo``` folder.

### Preprocessing demo

This demo can be run using the script ```demo_preprocessing.sh```.
Input data are in ```demo/scan```, output data are saved in ```demo/inversion```.


### Visualization

The input and output data in the ```demo/inversion``` folder can be visualized using the script ```src/View/RenderRecon/run.m```.
The file ```src/Modular_MatlabRoutines/read3D.m``` can be used to read the reconstruction files into MATLAB.

## Dependencies

#### 1) Bash
#### 2) GNU readlink 
Readlink converts relative paths to absolute paths.
Typically present in HPC clusters and GNU Linux machines but not Mac.
In Mac there is another readlink that has different behaviour than GNU readlink.

Check if readlink works by running: 
```
readlink -f ..
```
	
If readlink does not work, install it using homebrew(https://brew.sh) by doing:

```
brew install coreutils
alias readlink='greadlink'
```

Uncomment the code ```alias readlink='greadlink'``` in the run script ```run/basic_pipeline.sh``` to have GNU readlink present while running the reconstruction.

#### 3) Intel C compiler (ICC) with openmp support
For HPC clusters, run
```
module load intel
```
Otherwise, install it from intel's website
