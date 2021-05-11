
__lib_path = os.path.join(os.path.expanduser('~'), '.cache', 'mbircone')

def recon(sino, angles, dist_source_detector, magnification,
    center_offset=(0.0,0.0), rotation_offset=0.0, delta_pixel_detector=1.0, delta_pixel_image=None,
    num_rows=None, num_cols=None, num_slices=None, roi_radius=None,
    init_image=0.0, prox_image=None,
    sigma_y=None, snr_db=30.0, weights=None, weight_type='unweighted',
    positivity=True, p=1.2, q=2.0, T=1.0, num_neighbors=6,
    sharpness=0.0, sigma_x=None, max_iterations=20, stop_threshold=0.0,
    num_threads=None, NHICD=False, verbose=1, lib_path=__lib_path):

    """Computes 3D cone beam MBIR reconstruction
    
    Args:
        sino (ndarray): 3D sinogram array with shape (num_views, num_slices, num_channels)
        angles (ndarray): 1D view angles array in radians.
        dist_source_detector (float): Distance between the X-ray source and the detector in units of ALU
        magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
        
        center_offset (float tuple, optional): [Default=(0.0, 0.0)] Distance between detector center and center of beam in units of ALU.
            The first element of the tuple is the horizontal distance and the second element vertical.
        rotation_offset (float, optional): [Default=0.0] Distance between the projected center of rotation on the detector and the center of the detector in units of :math:`ALU`.
        delta_pixel_detector (float, optional): [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.
        delta_pixel_image (float, optional): [Default=None] Scalar value of image pixel spacing in :math:`ALU`.
            If None, automatically set to delta_pixel_detector/magnification
        num_rows (int, optional): [Default=None] Integer number of rows in reconstructed image.
            If None, automatically set.
        num_cols (int, optional): [Default=None] Integer number of columns in reconstructed image.
            If None, automatically set.
        num_slices (int, optional): [Default=None] Integer number of slices in reconstructed image.
            If None, automatically set.
        roi_radius (float, optional): [Default=None] Scalar value of radius of reconstruction in :math:`ALU`.
            If None, automatically set with auto_roi_radius().
            Pixels outside the radius roi_radius in the :math:`(x,y)` plane are disregarded in the reconstruction.
        
        init_image (ndarray, optional): [Default=0.0] Initial value of reconstruction image, specified by either a scalar value or a 3D numpy array with shape (num_slices,num_rows,num_cols)
        prox_image (ndarray, optional): [Default=None] 3D proximal map input image. 3D numpy array with shape (num_slices,num_rows,num_cols)
        
        sigma_y (float, optional): [Default=None] Scalar value of noise standard deviation parameter.
            If None, automatically set with auto_sigma_y.
        snr_db (float, optional): [Default=30.0] Scalar value that controls assumed signal-to-noise ratio of the data in dB.
            Ignored if sigma_y is not None.
        weights (ndarray, optional): [Default=None] 3D weights array with same shape as sino.
        weight_type (string, optional): [Default='unweighted'] Type of noise model used for data.
            If the ``weights`` array is not supplied, then the function ``mbircone.calc_weights`` is used to set weights using specified ``weight_type`` parameter.
            Option "unweighted" corresponds to unweighted reconstruction;
            Option "transmission" is the correct weighting for transmission CT with constant dosage;
            Option "transmission_root" is commonly used with transmission CT data to improve image homogeneity;
            Option "emission" is appropriate for emission CT data.

        positivity (bool, optional): [Default=True] Boolean value that determines if positivity constraint is enforced. 
            The positivity parameter defaults to True; however, it should be changed to False when used in applications that can generate negative image values.
        p (float, optional): [Default=1.2] Scalar value in range :math:`[1,2]` that specifies the qGGMRF shape parameter.
        q (float, optional): [Default=2.0] Scalar value in range :math:`[p,1]` that specifies the qGGMRF shape parameter.
        T (float, optional): [Default=1.0] Scalar value :math:`>0` that specifies the qGGMRF threshold parameter.
        num_neighbors (int, optional): [Default=6] Possible values are {26,18,6}.
            Number of neightbors in the qggmrf neighborhood. Higher number of neighbors result in a better regularization but a slower reconstruction.
        sharpness (float, optional): [Default=0.0]
            Scalar value that controls level of sharpness in the reconstruction
            ``sharpness=0.0`` is neutral; ``sharpness>0`` increases sharpness; ``sharpness<0`` reduces sharpness.
            Ignored if sigma_x is not None.
        sigma_x (float, optional): [Default=None] Scalar value :math:`>0` that specifies the qGGMRF/proxmap scale parameter.
            If None, automatically set with auto_sigma_x. The parameter sigma_x can be used to directly control regularization, but this is only recommended for expert users.
        max_iterations (int, optional): [Default=20] Integer valued specifying the maximum number of iterations. 
        stop_threshold (float, optional): [Default=0.0] [Default=0.02] Scalar valued stopping threshold in percent.
            If stop_threshold=0.0, then run max iterations.
        num_threads (int, optional): [Default=None] Number of compute threads requested when executed.
            If None, num_threads is set to the number of cores in the system
        NHICD (bool, optional): [Default=False] If true, uses Non-homogeneous ICD updates
        verbose (int, optional): [Default=1] Possible values are {0,1,2}, where 0 is quiet, 1 prints minimal reconstruction progress information, and 2 prints the full information.
        lib_path (str, optional): [Default=~/.cache/mbircone] Path to directory containing library of forward projection matrices.
    Returns:
        3D numpy array: 3D reconstruction with shape (num_slices,num_rows,num_cols) in units of :math:`ALU^{-1}`.
    """

    # Internally set
    # NHICD_ThresholdAllVoxels_ErrorPercent=80, NHICD_percentage=15, NHICD_random=20, 
    # zipLineMode=2, N_G=2, numVoxelsPerZiplineMax=200
    
    pass

