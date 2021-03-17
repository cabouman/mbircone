
Strategy:

1. Define target C subroutine interface
2. Modify C code to conformed to target interface (make wrapper + other changes)
3. Write cython wrapper to call C subroutines
4. Write python wrapper to using cython subroutines


```
recon(sino, wght, x_init, proxmap_input,
	N_dv, N_dw, N_beta, Delta_dv, Delta_dw, u_s, u_r, v_r, u_d0, v_d0, w_d0, TotalAngle, 

	reconparams, py_Amatrix_fname):

```


TO DO:

clean reconparams
Add verbose levels

add function to convert ROI to ROR


============================================


- Write python preprocessing: read radiograph tif from fodler and convert to sino np ndarray
    - Demo with radiographs (metal-weld data)

1) Write preprocess_conebeam function
2) Visulaize output
3) Preprocess metal-weld data with python

4) Preprocess metal-weld params with command line
5) Reconstruct with data (from python preprocessing) amd params (from comamnd line preprocessing)

- Simply recon inputs


Recon params:
```


################################################################ Prior Model stuff #########
[priorWeight_QGGMRF] Weight of [1/2 ||y-Ax||^2_W] term. Skips computation if weight < 0. 
1

[priorWeight_proxMap] Weight of [1/(2 sigma_lambda^2) ||x-x~||^2] term. Skips computation if weight < 0. 
-1

[is_positivity_constraint] (1: positivity constraint on, 0: positivity constraint off )
1
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . QGGMRF . . . . . . . . . . .

[q] QGGMRF parameter (q>1, typical choice q=2)
2

[p] QGGMRF parameter (1<=p<q)
1

[T] QGGMRF parameter 		(=eps TGGMRF parameter)
0.02

[sigmaX] QGGMRF parameter	(=s TGGMRF parameter)
5

num_neighbors

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . Proximal Mapping . . . . . .
[sigma_lambda] Proximal mapping scaler
1


[MaxIterations] maximum number of iterations
10

[zipLineMode] (0: off, 1: conventional Zipline, 2: randomized Zipline, 3: Super Voxel ZL)
2

[N_G] Number of groups for group ICD
2

[numVoxelsPerZiplineMax] ziplines will have ceil(N_z/ceil(N_z/#)) voxels
200

################################################################ Parallel stuff ############
[numThreads] Number of threads
20
################################################################ Weight Scaler  ############
[weightScaler_domain] "spatiallyVariant", "spatiallyInvariant"
spatiallyInvariant
spatiallyVariant

[weightScaler_estimateMode] 'None', "errorSino", "avgWghtRecon" (only when "spatiallyInvariant")
avgWghtRecon
None
errorSino

[weightScaler_value] User specified weight scaler (only when 'spatiallyInvariant' and 'None')
1
################################################################ NHICD          ############
[NHICD_Mode] 'off' 'percentile+random'
off

[NHICD_ThresholdAllVoxels_ErrorPercent] when error greater then all voxels are updated
80

[NHICD_percentage] lastChange>prctile(lastChange, 100-#) is updated
15

[NHICD_random] approx #% of remaining voxels are updated randomly
20

################################################################ Misc           ############
[verbosity] 0: minimal output, 1: medium output, 2: more, 3: maximum output
0

[isComputeCost]
0


```
