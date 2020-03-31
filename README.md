# OpenMBIR-ConeBeam
Open Source Cone Beam MBIR package

Distribution Statement A. Approved for public release: distribution unlimited (88ABW-2020-0895).
If you publish results based on this code, please cite the following paper:
> Thilo Balke, Soumendu Majee, Gregery T. Buzzard, Scott Poveromo, Patrick Howard, Michael A. Groeber, John McClure, Charles A. Bouman "Separable Models for cone-beam MBIR Reconstruction," proceedings of the IS&T International Symposium on Electronic Imaging, Computational Imaging XVI, pp. 181-1 to 181-7, 2018.

For Parallel Beam MBIR, please see:
https://github.com/cabouman/OpenMBIR-ParBeam

For Super-Voxel (i.e., fast parallelized) Parallel Beam MBIR, please see
https://github.com/HPImaging/sv-mbirct

## Organization

The codebase is broadly divided into five parts:
1) binaries: meant to contain all binary data files: (reconstructions, sinogram etc)
2) code: the actual source code
3) control: contains parameters inputted to the source code (not included)
4) run: contains run scripts for all major parts of the code
5) template_control: a reference 'control' folder containing ‘default’ parameter values

## Running

First create a 'control' folder and copy the contents of 'template_control' into the 'control' folder.
The parameters in 'control' can be modified as needed.

Then, run the basic reconstruction pipeline given in run/basic_pipeline.sh

To submit a SLURM job in a cluster with the basic pipeline, run 
```
sbatch jobSub/SLURM_CB.sub
```
from the run directory


## Dependencies


#### 1) readlink 
	Present in HPC clusters and GNU Linux machines but not Mac
	Check if readlink works: 
		```
		readlink -f ..
		```
	If not, run:
		```
		brew install coreutils
		alias readlink='greadlink'
		```
	Add the alias code snippet to the bash_profile to have it always present.

	
#### 2) Bash

#### 3) Intel C compiler (ICC) with openmp support
	For HPC clusters, run
	```
	module load intel
	```

#### 4) Matlab with command line support
	For HPC clusters, run
	```
	module load matlab
	```

	For standalone computers, run the following code to add command line support for matlab
	```
	alias matlab="/Applications/MATLAB_R2018b.app/bin/matlab -nojvm -nodesktop"
	```
	[change the path to wherever matlab is installed in the system.]
	Add the alias code snippet to the bash_profile to have it always present.

	Follow the full directions in: https://www.mathworks.com/matlabcentral/answers/442969-how-do-i-run-matlab-from-the-mac-terminal





