# OpenMBIR-ConeBeam
Open Source Cone Beam MBIR package

Distribution Statement A. Approved for public release: distribution unlimited (88ABW-2020-0895).
If you publish results based on this code, please cite the following paper:
> Thilo Balke, Soumendu Majee, Gregery T. Buzzard, Scott Poveromo, Patrick Howard, Michael A. Groeber, John McClure, Charles A. Bouman "Separable Models for cone-beam MBIR Reconstruction," proceedings of the IS&T International Symposium on Electronic Imaging, Computational Imaging XVI, pp. 181-1 to 181-7, 2018.

For other OpenMBIR packages see: https://github.com/cabouman/OpenMBIR-Index

## Organization

The codebase is broadly divided into five parts:
1) bin: meant to store compiled executables
2) data: meant to contain all data files: (scans, reconstructions, sinogram etc)
3) param: contains parameters for reconstruction
4) run: contains scripts that compile and execute the code
5) src: the actual source code

## Running

To reconstruct from the included data in ```data/Scan```, run the script in ```run/basic_pipeline.sh``` from the run folder.

To submit a SLURM job in a HPC cluster with the basic pipeline, run ```sbatch jobSub/SLURM_CB.sub``` from the run folder.

## Data and Visualization

The repository includes a small test data in ```data/Scan```.
The script ```run/basic_pipeline.sh``` reconstructs from this test data.

The reconstructed images are stored in ```data/ConeBeam```.
The data can be visualized using the script ```src/View/RenderRecon/run.m```
A volume rendering can also be performed by running ```src/View/VolumeRenderRecon/run.m```
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
