
def recon(sino, wght, angles, x_init, proxmap_input, Amatrix_fname,
	detector_loc_horz, detector_loc_vert, dist_source_obj, dist_detector_obj,
	delta_pixel_detector, delta_pixel_image, rotation_offset,
	is_QGGMRF, is_proxMap, is_positivity_constraint,
	q, p, T, sigma_x, num_neighbors, sigma_proxmap, max_iterations,
	zipLineMode, N_G, numVoxelsPerZiplineMax, numThreads, 
	weightScaler_domain, weightScaler_estimateMode, weightScaler_value,
	NHICD_Mode, NHICD_ThresholdAllVoxels_ErrorPercent, NHICD_percentage, NHICD_random, 
	verbosity, isComputeCost):
    """Summary
    
    Args:
        sino (ndarray): 3D sinogram array with shape (num_views, num_slices, num_channels)
        wght (ndarray): 3D weights array with same shape as sino.
        angles (ndarray): 1D view angles array in radians.
        x_init (ndarray): Shape (num_slices,num_rows,num_cols)
        proxmap_input (ndarray): Shape (num_slices,num_rows,num_cols)
        Amatrix_fname (string): Filename of system matrix
        
        detector_offset_horz (float): Detector offset in horz direction
        detector_offset_vert (float): Detector offset in vert direction
        dist_source_obj (float): Source to object distance
        dist_detector_obj (float): Detector to object distance
        delta_pixel_detector (float): Pixel pitch in the detector
        delta_pixel_image (float): Pixel pitch in the reconstruction
        rotation_offset_u (float): Location of rotation axis wrt center of object along u axis
        rotation_offset_v (float): Location of rotation axis wrt center of object along v axis
        

        is_QGGMRF (float): 
        is_proxMap (float): 
        is_positivity_constraint (float): 
        q (float): 
        p (float): 
        T (float): 
        sigma_x (float):
        sigma_proxmap (float):
        num_neighbors (int): Number of neighbors in qggmrf prior
        max_iterations (int): 
        zipLineMode (int): 
        N_G (int): Number of groups for group ICD
        numVoxelsPerZiplineMax (int): Max zipline size
        numThreads (int): Number of threads
        weightScaler_domain (string): 
        weightScaler_estimateMode (string): 
        weightScaler_value (float): 
        NHICD_Mode (string): 
        NHICD_ThresholdAllVoxels_ErrorPercent (float): 
        NHICD_percentage (float): 
        NHICD_random (float): 
        verbosity (int): 
        isComputeCost (int): 
    """
