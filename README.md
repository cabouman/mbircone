# OpenMBIR-ConeBeam
Open Source Cone Beam MBIR package

Distribution Statement A. Approved for public release: distribution unlimited (88ABW-2020-0895).
If you publish results based on this code, please cite the following paper:
> Thilo Balke, Soumendu Majee, Gregery T. Buzzard, Scott Poveromo, Patrick Howard, Michael A. Groeber, John McClure, Charles A. Bouman "Separable Models for cone-beam MBIR Reconstruction," proceedings of the IS&T International Symposium on Electronic Imaging, Computational Imaging XVI, pp. 181-1 to 181-7, 2018.

For other OpenMBIR packages see: https://github.com/cabouman/OpenMBIR-Index

## Organization

The codebase is broadly divided into five parts:
1) data: meant to contain all binary data files: (reconstructions, sinogram etc)
2) src: the actual source code
3) control: contains parameters inputted to the source code (not included)
4) run: contains run scripts for all major parts of the code
5) template_control: a reference 'control' folder containing ‘default’ parameter values

## Running

First create a ```control``` folder and copy the contents of ```template_control``` into the ```control``` folder.
The parameters in ```control``` can be modified as needed.

Then, run the basic reconstruction pipeline given in ```run/basic_pipeline.sh```

To submit a SLURM job in a HPC cluster with the basic pipeline, run 
```
sbatch jobSub/SLURM_CB.sub
```
from the run directory

## Data and Visualization

The repository includes a small test data in ```binaries/Scan```
The demo script ```run/basic_pipeline.sh``` reconstructs from this test data.

The reconstructed images are stored in ```binaries/ConeBeam```.
The data can be visualized using the script ```code/View/RenderRecon/run.m```
A volume rendering can also be performed by running ```code/View/VolumeRenderRecon/run.m```

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
