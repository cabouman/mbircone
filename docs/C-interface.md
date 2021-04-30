
Strategy:

1. Define target C subroutine interface
2. Modify C code to conformed to target interface (make wrapper + other changes)
3. Write cython wrapper to call C subroutines
4. Write python wrapper to using cython subroutines


```
recon(sino, wght, angles, x_init, proxmap_input, Amatrix_fname,
	detector_loc_horz, detector_loc_vert, dist_source_obj, dist_detector_obj,
	delta_pixel_detector, delta_pixel_image, rotation_offset=[0,0],

	is_QGGMRF, is_proxMap, is_positivity_constraint
	q, p, T, sigma_x, num_neighbors, sigma_proxmap, max_iterations,

	zipLineMode, N_G, numVoxelsPerZiplineMax, numThreads, 
	weightScaler_domain, weightScaler_estimateMode, weightScaler_value,
	NHICD_Mode, NHICD_ThresholdAllVoxels_ErrorPercent, NHICD_percentage, NHICD_random, 
	verbosity, isComputeCost):


```


TO DO:

clean reconparams
Add verbose levels

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
