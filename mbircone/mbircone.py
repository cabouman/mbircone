
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
    channel_offset=0.0, row_offset=0.0, rotation_offset=0.0, delta_pixel_detector=1.0, delta_pixel_image=None):
    """ Computes sinogram parameters required by the Cython code
    
    Args:
        dist_source_detector (float): Distance between the X-ray source and the detector in units of ALU
        magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
        
        channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a row.
        row_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a column.
        rotation_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from source-detector line to axis of rotation in the object space.
            This is normally set to zero.
        delta_pixel_detector (float, optional): [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.
        delta_pixel_image (float, optional): [Default=None] Scalar value of image pixel spacing in :math:`ALU`.
            If None, automatically set to delta_pixel_detector/magnification
    
    Returns:
        Dictionary containing sino parameters as required by the Cython code
    """
    
    pass



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

    pass


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
    
    sinoparams = dict()
    sinoparams['N_dv'] = sino.shape[2]
    sinoparams['N_dw'] = sino.shape[1]
    sinoparams['N_beta'] = sino.shape[0]
    sinoparams['Delta_dv'] = 3.2
    sinoparams['Delta_dw'] = 3.2
    sinoparams['u_s'] = -71.6364
    sinoparams['u_r'] = 0
    sinoparams['v_r'] = 0
    sinoparams['u_d0'] = 548.1538
    sinoparams['v_d0'] = -101.4616
    sinoparams['w_d0'] = -102.9654
    sinoparams['weightScaler_value'] = -1

    imgparams = dict()
    imgparams['x_0'] = -11.9663
    imgparams['y_0'] = -11.9663
    imgparams['z_0'] = -14.7378
    imgparams['N_x'] = 65
    imgparams['N_y'] = 65
    imgparams['N_z'] = 80
    imgparams['Delta_xy'] = 0.36986
    imgparams['Delta_z'] = 0.36986
    imgparams['j_xstart_roi'] = 2
    imgparams['j_ystart_roi'] = 2
    imgparams['j_zstart_roi'] = 14
    imgparams['j_xstop_roi'] = 62
    imgparams['j_ystop_roi'] = 62
    imgparams['j_zstop_roi'] = 66
    imgparams['N_x_roi'] = 61
    imgparams['N_y_roi'] = 61
    imgparams['N_z_roi'] = 53

    reconparams = dict()
    reconparams['InitVal_recon'] = 0
    reconparams['initReconMode'] = 'constant'
    reconparams['priorWeight_QGGMRF'] = 1
    reconparams['priorWeight_proxMap'] = -1
    reconparams['is_positivity_constraint'] = 1
    reconparams['q'] = 2
    reconparams['p'] = 1
    reconparams['T'] = 0.02
    reconparams['sigmaX'] = 5
    reconparams['bFace'] = 1.0
    reconparams['bEdge'] = 0.70710678118
    reconparams['bVertex'] = 0.57735026919
    reconparams['sigma_lambda'] = 1
    reconparams['stopThresholdChange_pct'] = 0.00
    reconparams['stopThesholdRWFE_pct'] = 0
    reconparams['stopThesholdRUFE_pct'] = 0
    reconparams['MaxIterations'] = 10
    reconparams['relativeChangeMode'] = 'percentile'
    reconparams['relativeChangeScaler'] = 0.1
    reconparams['relativeChangePercentile'] = 99.9
    reconparams['zipLineMode'] = 2
    reconparams['N_G'] = 2
    reconparams['numVoxelsPerZiplineMax'] = 200
    reconparams['numVoxelsPerZipline'] = 200
    reconparams['numZiplines'] = 4
    reconparams['numThreads'] = 20
    reconparams['weightScaler_domain'] = 'spatiallyInvariant'
    reconparams['weightScaler_estimateMode'] = 'avgWghtRecon'
    reconparams['weightScaler_value'] = 1
    reconparams['NHICD_Mode'] = 'off'
    reconparams['NHICD_ThresholdAllVoxels_ErrorPercent'] = 80
    reconparams['NHICD_percentage'] = 15
    reconparams['NHICD_random'] = 20
    reconparams['verbosity'] = 0
    reconparams['isComputeCost'] = 0
    reconparams['backprojlike_type'] = 'proj'



    Amatrix_fname = 'test.sysmatrix'
    ci.AmatrixComputeToFile_cy(angles, sinoparams, imgparams, Amatrix_fname, verbose=1)
    x_init = np.zeros((imgparams['N_x'],imgparams['N_y'],imgparams['N_z']))
    proxmap_input = np.zeros((imgparams['N_x'],imgparams['N_y'],imgparams['N_z']))


    print('Reconstructing ...')
    sino = np.swapaxes(sino, 1, 2)
    weights = np.swapaxes(weights, 1, 2)
    x = ci.recon_cy(sino, weights, x_init, proxmap_input,
                 sinoparams, imgparams, reconparams, Amatrix_fname)
    print('Reconstructing done.')

    x = np.swapaxes(x, 0, 2)

    return x

