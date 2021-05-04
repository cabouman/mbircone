
__lib_path = os.path.join(os.path.expanduser('~'), '.cache', 'mbircone')

def recon(sino, angles, dist_source_detector, magnification,
    center_offset=(0.0,0.0), rotation_offset=0.0, delta_pixel_detector=1.0, delta_pixel_image=1.0,
    init_image=0.0, prox_image=None,
    sigma_y=None, snr_db=30.0, weights=None, weight_type='unweighted',
    is_qggmrf=True, is_proxmap=False, positivity=True, 
    q=2.0, p=1.2, T=2.0, num_neighbors=26,
    sigma_x=None, sigma_proxmap=None, max_iterations=20, stop_threshold=0.0,
    num_threads=None, 
    is_NHICD=False,
    verbose=False,
    lib_path=__lib_path):
    """Computes 3D cone beam MBIR reconstruction
    
    Args:
        sino (ndarray): 3D sinogram array with shape (num_views, num_slices, num_channels)
        angles (ndarray): 1D view angles array in radians.
        dist_source_detector (float): Distance between the X-ray source and the detector in units of ALU
        magnification (float): Magnification of the cone-beam geometry
        center_offset (float tuple, optional): Distance between detector center and center of beam in units of ALU.
            Thhe first element of the tuple is the horizontal distance and the secon element vertical.
        rotation_offset (float, optional): Distance between object center and axis of rotatation in units of ALU
        delta_pixel_detector (float, optional): Scalar value of detector pixel spacing in :math:`ALU`.
        delta_pixel_image (float, optional): Scalar value of image pixel spacing in :math:`ALU`.
        init_image (ndarray, optional): Initial value of reconstruction image, specified by either a scalar value or a 3D numpy array with shape (num_slices,num_rows,num_cols)
        prox_image (ndarray, optional): 3D proximal map input image. 3D numpy array with shape (num_slices,num_rows,num_cols)
        sigma_y (float, optional): Scalar value of noise standard deviation parameter.
            If None, automatically set with auto_sigma_y.
        snr_db (float, optional): Scalar value that controls assumed signal-to-noise ratio of the data in dB.
            Ignored if sigma_y is not None.
        weights (ndarray, optional): 3D weights array with same shape as sino.
        weight_type (string, optional): Type of noise model used for data.
            If the ``weights`` array is not supplied, then the function ``svmbir.calc_weights`` is used to set weights using specified ``weight_type`` parameter.
            Option "unweighted" corresponds to unweighted reconstruction;
            Option "transmission" is the correct weighting for transmission CT with constant dosage;
            Option "transmission_root" is commonly used with transmission CT data to improve image homogeneity;
            Option "emission" is appropriate for emission CT data.

        is_qggmrf (bool, optional): Boolean value that determines if qggmrf prior is enables.
            If false, the reconstruction is a proximal operator if prox_image is provided, else it is unregularized.
        positivity (bool, optional): Boolean value that determines if positivity constraint is enforced. 
            The positivity parameter defaults to True; however, it should be changed to False when used in applications that can generate negative image values.
        p (float, optional): Scalar value in range :math:`[1,2]` that specifies the qGGMRF shape parameter.
        q (float, optional): Scalar value in range :math:`[p,1]` that specifies the qGGMRF shape parameter.
        T (float, optional): Scalar value :math:`>0` that specifies the qGGMRF threshold parameter.
        num_neighbors (int, optional):Possible values are {26,18,6}.  Number of neightbors in the qggmrf neighborhood. 
        sigma_x (float, optional): Scalar value :math:`>0` that specifies the qGGMRF scale parameter.
            If None, automatically set with auto_sigma_x. The parameter sigma_x can be used to directly control regularization, but this is only recommended for expert users.
        sigma_proxmap (float, optional): Scalar value :math:`>0` that specifies the proximal map scale parameter.
        max_iterations (int, optional): Integer valued specifying the maximum number of iterations. 
        stop_threshold (float, optional): [Default=0.02] Scalar valued stopping threshold in percent.
            If stop_threshold=0.0, then run max iterations.
        num_threads (int, optional): Number of compute threads requested when executed.
            If None, num_threads is set to the number of cores in the system
        is_NHICD (bool, optional): If true, uses Non-homogeneous ICD updates
        verbose (bool, optional): If true, displays reconstruction progress information
        lib_path (str, optional): Path to directory containing library of forward projection matrices.
    Returns:
        3D numpy array: 3D reconstruction with shape (num_slices,num_rows,num_cols) in units of :math:`ALU^{-1}`.
    """

    # Internally set
    # NHICD_ThresholdAllVoxels_ErrorPercent=80, NHICD_percentage=15, NHICD_random=20, 
    # zipLineMode=2, N_G=2, numVoxelsPerZiplineMax=200
    
    pass

