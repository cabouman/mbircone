
import math
from psutil import cpu_count
import shutil
import numpy as np
import os
import mbircone.interface_cy_c as ci

__lib_path = os.path.join(os.path.expanduser('~'), '.cache', 'mbircone')

def _mbircone_lib_path():
    """Returns the path to the cache directory used by mbircone
    """
    return __lib_path


def _clear_cache(mbircone_lib_path=__lib_path):
    """Clears the cache files used by mbircone
    
    Args:
        mbircone_lib_path (string): Path to mbircone cache directory. Defaults to __lib_path variable
    """
    shutil.rmtree(mbircone_lib_path)


def calc_weights(sino, weight_type):
    """Computes the weights used in MBIR reconstruction.

    Args:
        sino (ndarray): 3D numpy array of sinogram data with shape (num_views,num_slices,num_channels)
        weight_type (string):[Default=0] Type of noise model used for data.

            If weight_type="unweighted"        => weights = numpy.ones_like(sino)

            If weight_type="transmission"      => weights = numpy.exp(-sino)

            If weight_type="transmission_root" => weights = numpy.exp(-sino/2)

            If weight_type="emission"         => weights = 1/(sino + 0.1)

    Returns:
        ndarray: 3D numpy array of weights with same shape as sino.

    Raises:
        Exception: Description
    """
    if weight_type == 'unweighted' :
        weights = np.ones(sino.shape)
    elif weight_type == 'transmission' :
        weights = np.exp(-sino)
    elif weight_type == 'transmission_root' :
        weights = np.exp(-sino / 2)
    elif weight_type == 'emission' :
        weights = 1 / (sino + 0.1)
    else :
        raise Exception("calc_weights: undefined weight_type {}".format(weight_type))

    return weights


def auto_sigma_y(sino, weights, snr_db=30.0, delta_pixel_image=1.0, delta_pixel_detector=1.0):
    """Computes the automatic value of ``sigma_y`` for use in MBIR reconstruction.

    Args:
        sino (ndarray):
            3D numpy array of sinogram data with shape (num_views,num_slices,num_channels)
        weights (ndarray):
            3D numpy array of weights with same shape as sino.
            The parameters weights should be the same values as used in mbircone reconstruction.
        snr_db (float, optional):
            [Default=30.0] Scalar value that controls assumed signal-to-noise ratio of the data in dB.
        delta_pixel_image (float, optional):
            [Default=1.0] Scalar value of pixel spacing in :math:`ALU`.
        delta_pixel_detector (float, optional):
            [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.


    Returns:
        ndarray: Automatic values of regularization parameter.
    """
    
    pass


def auto_sigma_x(sino, delta_pixel_detector=1.0, sharpness=0.0):
    """Computes the automatic value of ``sigma_x`` for use in MBIR reconstruction.

    Args:
        sino (ndarray):
            3D numpy array of sinogram data with shape (num_views,num_slices,num_channels)
        delta_pixel_detector (float, optional):
            [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.
        sharpness (float, optional):
            [Default=0.0] Scalar value that controls level of sharpness.
            ``sharpness=0.0`` is neutral; ``sharpness>0`` increases sharpness; ``sharpness<0`` reduces sharpness

    Returns:
        float: Automatic value of regularization parameter.
    """
    (num_views, num_slices, num_channels) = sino.shape

    # Compute indicator function for sinogram support
    sino_indicator = _sino_indicator(sino)

    # Compute a typical image value by dividing average sinogram value by a typical projection path length
    typical_img_value = np.average(sino, weights=sino_indicator) / (num_channels * delta_pixel_detector)

    # Compute sigma_x as a fraction of the typical image value
    sigma_x = 0.2 * (2 ** sharpness) * typical_img_value

    return sigma_x


def compute_sino_params(dist_source_detector, magnification, 
    num_views, num_slices, num_channels,
    channel_offset=0.0, row_offset=0.0, rotation_offset=0.0, 
    delta_pixel_detector=1.0):
    """ Computes sinogram parameters required by the Cython code
    
    Args:
        dist_source_detector (float): Distance between the X-ray source and the detector in units of ALU
        magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
        
        num_views (int): Number of views in sinogram data
        num_slices (int): Number of slices in sinogram data
        num_channels (int): Number of channels in sinogram data

        channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a row.
        row_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a column.
        rotation_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from source-detector line to axis of rotation in the object space.
            This is normally set to zero.
        delta_pixel_detector (float, optional): [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.
    
    Returns:
        Dictionary containing sino parameters as required by the Cython code
    """
    
    sinoparams = dict()
    sinoparams['N_dv'] = num_channels
    sinoparams['N_dw'] = num_slices
    sinoparams['N_beta'] = num_views
    sinoparams['Delta_dv'] = delta_pixel_detector
    sinoparams['Delta_dw'] = delta_pixel_detector
    sinoparams['u_s'] = - dist_source_detector/magnification
    sinoparams['u_r'] = 0
    sinoparams['v_r'] = rotation_offset
    sinoparams['u_d0'] = dist_source_detector - dist_source_detector/magnification


    dist_dv_to_detector_corner_from_detector_center = - sinoparams['N_dv']*sinoparams['Delta_dv']/2
    dist_dw_to_detector_corner_from_detector_center = - sinoparams['N_dw']*sinoparams['Delta_dw']/2

    dist_dv_to_detector_center_from_source_detector_line = - channel_offset
    dist_dw_to_detector_center_from_source_detector_line = - row_offset

    # corner of detector from source-detector-line
    sinoparams['v_d0'] = dist_dv_to_detector_corner_from_detector_center + dist_dv_to_detector_center_from_source_detector_line
    sinoparams['w_d0'] = dist_dw_to_detector_corner_from_detector_center + dist_dw_to_detector_center_from_source_detector_line

    sinoparams['weightScaler_value'] = -1

    return sinoparams



def compute_img_params(sinoparams, delta_pixel_image=None,
    num_rows=None, num_cols=None, num_slices=None, roi_radius=None):
    """ Computes image parameters required by the Cython code
    
    Args:
        sinoparams (dict): Dictionary containing sinogram parameters as required by the Cython code
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
    
    Returns:
        Dictionary containing image parameters as required by the Cython code
     
    """
    # port code from https://github.com/cabouman/OpenMBIR-ConeBeam/blob/fbf3eddcadad1bd1cfb657f58c5b32f1204a12d1/utils/Preprocessing/Modular_PreprocessingRoutines/computeImgParams.m


    # Part 1: find radius of circle
    # lower cone point P0
    P0 = (sinoparams['u_d0'], sinoparams['v_d0'])

    # upper cone point P1
    P1 = (sinoparams['u_d0'], sinoparams['v_d0'] + sinoparams['N_dv']*sinoparams['Delta_dv']);

    # source point S
    S = (sinoparams['u_s'], 0);

    # Rotation center point C
    C = (sinoparams['u_r'], sinoparams['v_r']);

    # r_0 = distance{ line(P0,S), C }
    r_0 = _distance_line_to_point( P0, S, C)

    # r_1 = distance{ line(P1,S), C }
    r_1 = _distance_line_to_point( P1, S, C)

    r = max(r_0, r_1);


    # #### Part 2: assignment of parameters ####

    imgparams = dict()
    imgparams['Delta_xy'] = delta_pixel_image
    imgparams['Delta_z'] = delta_pixel_image

    imgparams['x_0'] = -(r + imgparams['Delta_xy']/2)
    imgparams['y_0'] = imgparams['x_0']

    imgparams['N_x'] = 2*math.ceil( r / imgparams['Delta_xy']) + 1
    imgparams['N_y'] = imgparams['N_x']


    ## Computation of z_0 and N_z 

    x_1 = imgparams['x_0'] + imgparams['N_x']*imgparams['Delta_xy']
    y_1 = x_1

    R_00 = math.sqrt(imgparams['x_0']**2 + imgparams['y_0']**2)
    R_10 = math.sqrt(x_1**2 + imgparams['y_0']**2)
    R_01 = math.sqrt(imgparams['x_0']**2 + y_1**2)
    R_11 = math.sqrt(x_1**2 + y_1**2)

    R = max(R_00, R_10, R_01, R_11)

    w_1 = sinoparams['w_d0'] + sinoparams['N_dw']*sinoparams['Delta_dw']


    z_0 = min(  sinoparams['w_d0'] * ( R - sinoparams['u_s']) / (sinoparams['u_d0'] - sinoparams['u_s']), 
                sinoparams['w_d0'] * (-R - sinoparams['u_s']) / (sinoparams['u_d0'] - sinoparams['u_s']))

    z_1 = max(  w_1 * ( R - sinoparams['u_s']) / (sinoparams['u_d0'] - sinoparams['u_s']),
                w_1 * (-R - sinoparams['u_s']) / (sinoparams['u_d0'] - sinoparams['u_s']))

    imgparams['z_0'] = z_0
    imgparams['N_z'] = math.ceil(  (z_1-z_0)/(imgparams['Delta_z'])  )


    ## ROI parameters

    R_roi = min(r_0, r_1) - imgparams['Delta_xy'];

    w_0_roi = sinoparams['w_d0'] + sinoparams['Delta_dw'];
    w_1_roi = w_0_roi + (sinoparams['N_dw']-2)*sinoparams['Delta_dw'];

    z_min_roi = max(    w_0_roi * (-R_roi - sinoparams['u_s']) / (sinoparams['u_d0'] - sinoparams['u_s']),
                        w_0_roi * ( R_roi - sinoparams['u_s']) / (sinoparams['u_d0'] - sinoparams['u_s']))

    z_max_roi = min(    w_1_roi * (-R_roi - sinoparams['u_s']) / (sinoparams['u_d0'] - sinoparams['u_s']),
                        w_1_roi * ( R_roi - sinoparams['u_s']) / (sinoparams['u_d0'] - sinoparams['u_s']))

    N_x_roi = 2*math.floor(R_roi / imgparams['Delta_xy']) + 1
    N_y_roi = N_x_roi

    imgparams['j_xstart_roi'] = (imgparams['N_x'] - N_x_roi) / 2
    imgparams['j_ystart_roi'] = imgparams['j_xstart_roi']

    imgparams['j_xstop_roi'] = imgparams['j_xstart_roi'] + N_x_roi - 1
    imgparams['j_ystop_roi'] = imgparams['j_xstop_roi']

    imgparams['j_zstart_roi'] = round((z_min_roi - imgparams['z_0']) / imgparams['Delta_z'])
    imgparams['j_zstop_roi'] = imgparams['j_zstart_roi'] + round((z_max_roi-z_min_roi) / imgparams['Delta_z'])

    # Internally set by C code
    imgparams['N_x_roi'] = -1
    imgparams['N_y_roi'] = -1
    imgparams['N_z_roi'] = -1


    return imgparams



def _distance_line_to_point(A, B, P):
    """Computes the distance from point P to the line passing through points A and B
    
    Args:
        A (float, 2-tuple): (x,y) coordinate of point A
        B (float, 2-tuple): (x,y) coordinate of point B
        P (float, 2-tuple): (x,y) coordinate of point P
    """

    (x1,y1) = A
    (x2,y2) = B
    (x0,y0) = P

    # Line joining A,B has equation ax+by+c=0
    a = y2-y1
    b = -(x2-x1)
    c = y1*x2-x1*y2

    dist = abs(a*x0 + b*y0 + c)/math.sqrt(a**2 + b**2)

    return dist

# def auto_roi_radius():


def recon(sino, angles, dist_source_detector, magnification,
    channel_offset=0.0, row_offset=0.0, rotation_offset=0.0, delta_pixel_detector=1.0, delta_pixel_image=None,
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
        
        channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a row.
        row_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a column.
        rotation_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from source-detector line to axis of rotation in the object space.
            This is normally set to zero.
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


    if num_threads is None :
        num_threads = cpu_count(logical=False)
    
    (num_views, num_slices, num_channels) = sino.shape
    
    sinoparams = compute_sino_params(dist_source_detector, magnification,
    num_views=num_views, num_slices=num_slices, num_channels=num_channels,
    channel_offset=channel_offset, row_offset=row_offset, rotation_offset=rotation_offset, 
    delta_pixel_detector=delta_pixel_detector)

    imgparams = compute_img_params(sinoparams, delta_pixel_image=delta_pixel_image,
    num_rows=num_rows, num_cols=num_cols, num_slices=num_slices, roi_radius=roi_radius)

    reconparams = dict()

    reconparams['is_positivity_constraint'] = int(positivity)
    reconparams['q'] = q
    reconparams['p'] = p
    reconparams['T'] = T
    reconparams['sigmaX'] = sigma_x
    reconparams['bFace'] = 1.0
    reconparams['bEdge'] = 0.70710678118
    reconparams['bVertex'] = 0.57735026919
   
    reconparams['stopThresholdChange_pct'] = 0.00
    reconparams['stopThesholdRWFE_pct'] = 0
    reconparams['stopThesholdRUFE_pct'] = 0
    reconparams['MaxIterations'] = 10
    reconparams['relativeChangeMode'] = 'percentile'
    reconparams['relativeChangeScaler'] = 0.1
    reconparams['relativeChangePercentile'] = 99.9
    
    reconparams['weightScaler_domain'] = 'spatiallyInvariant'
    reconparams['weightScaler_estimateMode'] = 'avgWghtRecon'
    reconparams['weightScaler_value'] = 1
    
    reconparams['numThreads'] = num_threads
    reconparams['verbosity'] = verbose

    # Internally set
    reconparams['zipLineMode'] = 2
    reconparams['N_G'] = 2
    reconparams['numVoxelsPerZiplineMax'] = 200
    reconparams['numVoxelsPerZipline'] = 200
    reconparams['numZiplines'] = 4

    if NHICD:
        reconparams['NHICD_Mode'] = 'percentile+random'
    else:
        reconparams['NHICD_Mode'] = 'off'

    reconparams['NHICD_ThresholdAllVoxels_ErrorPercent'] = 80
    reconparams['NHICD_percentage'] = 15
    reconparams['NHICD_random'] = 20
    reconparams['isComputeCost'] = 1



    Amatrix_fname = 'test.sysmatrix'
    ci.AmatrixComputeToFile_cy(angles, sinoparams, imgparams, Amatrix_fname, verbose=verbose)


    if np.isscalar(init_image):
        init_image = np.zeros((imgparams['N_x'],imgparams['N_y'],imgparams['N_z']))+init_image
    
    if prox_image is None:
        reconparams['priorWeight_QGGMRF'] = 1
        reconparams['priorWeight_proxMap'] = -1
        proxmap_input = np.zeros((imgparams['N_x'],imgparams['N_y'],imgparams['N_z']))
        reconparams['sigma_lambda'] = 1
    else:
        reconparams['priorWeight_QGGMRF'] = -1
        reconparams['priorWeight_proxMap'] = 1
        reconparams['sigma_lambda'] = sigma_x
        proxmap_input = prox_image


    print('Reconstructing ...')
    sino = np.swapaxes(sino, 1, 2)
    weights = np.swapaxes(weights, 1, 2)
    x = ci.recon_cy(sino, weights, init_image, proxmap_input,
                 sinoparams, imgparams, reconparams, Amatrix_fname)
    print('Reconstructing done.')

    x = np.swapaxes(x, 0, 2)

    return x

