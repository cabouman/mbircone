import math
from psutil import cpu_count
import shutil
import numpy as np
import os
import hashlib
import mbircone.interface_cy_c as ci
import random
import warnings
import mbircone._utils as _utils

__lib_path = os.path.join(os.path.expanduser('~'), '.cache', 'mbircone')
__namelen_sysmatrix = 20


def _sino_indicator(sino):
    """ Compute a binary function that indicates the region of sinogram support.

    Args:
        sino (float, ndarray): Sinogram data with either 3D shape (num_views, num_det_rows, num_det_channels)
            or 4D shape (num_time_points, num_views, num_det_rows, num_det_channels).

    Returns:
        (int8, ndarray): Binary values corresponding to the support of ``sino``, with the same array shape as ``sino``.
            =1 within sinogram support; =0 outside sinogram support.
    """
    indicator = np.int8(sino > 0.05 * np.mean(np.fabs(sino)))  # for excluding empty space from average
    return indicator


def _distance_line_to_point(A, B, P):
    """ Compute the distance from point P to the line passing through points A and B. (Deprecated method)
    
    Args:
        A (float, 2-tuple): (x,y) coordinate of point A
        B (float, 2-tuple): (x,y) coordinate of point B
        P (float, 2-tuple): (x,y) coordinate of point P

    Returns:
        (float): Distance from point P to the line passing through points A and B.
    """

    (x1, y1) = A
    (x2, y2) = B
    (x0, y0) = P

    # Line joining A,B has equation ax+by+c=0
    a = y2 - y1
    b = -(x2 - x1)
    c = y1 * x2 - x1 * y2

    dist = abs(a * x0 + b * y0 + c) / math.sqrt(a ** 2 + b ** 2)

    return dist


def calc_weights(sino, weight_type):
    """ Compute the weights used in MBIR reconstruction.

    Args:
        sino (float, ndarray): Sinogram data with either 3D shape (num_views, num_det_rows, num_det_channels)
            or 4D shape (num_time_points, num_views, num_det_rows, num_det_channels).

        weight_type (string): Type of noise model used for data

                - weight_type = 'unweighted' => return numpy.ones(sino.shape).
                - weight_type = 'transmission' => return numpy.exp(-sino).
                - weight_type = 'transmission_root' => return numpy.exp(-sino/2).
                - weight_type = 'emission' => return 1/(numpy.absolute(sino) + 0.1).

    Returns:
        (float, ndarray): Weights used in mbircone reconstruction, with the same array shape as ``sino``.

    Raises:
        Exception: Raised if ``weight_type`` is not one of the above options.
    """
    if weight_type == 'unweighted':
        weights = np.ones(sino.shape)
    elif weight_type == 'transmission':
        weights = np.exp(-sino)
    elif weight_type == 'transmission_root':
        weights = np.exp(-sino / 2)
    elif weight_type == 'emission':
        weights = 1 / (np.absolute(sino) + 0.1)
    else:
        raise Exception("calc_weights: undefined weight_type {}".format(weight_type))

    return weights


def auto_max_resolutions(init_image) :
    """ Compute the automatic value of ``max_resolutions`` for use in MBIR reconstruction.

    Args:
        init_image (float, ndarray): Initial value of reconstruction image, specified by either a
            scalar value or a 3D numpy array with shape (num_img_slices, num_img_rows, num_img_cols).

    Returns:
        (int): Automatic value of ``max_resolutions``. Return ``0`` if ``init_image`` is a 3D numpy array,
            otherwise return ``2``.
    """
    # Default value of max_resolutions
    max_resolutions = 2
    if isinstance(init_image, np.ndarray) and (init_image.ndim == 3):
        #print('Init image present. Setting max_resolutions = 0.')
        max_resolutions = 0

    return max_resolutions


def auto_sigma_y(sino, magnification, weights, snr_db=40.0, delta_pixel_image=1.0, delta_pixel_detector=1.0):
    """ Compute the automatic value of ``sigma_y`` for use in MBIR reconstruction.

    Args:
        sino (float, ndarray): Sinogram data with either 3D shape (num_views, num_det_rows, num_det_channels)
            or 4D shape (num_time_points, num_views, num_det_rows, num_det_channels).
        magnification (float): Magnification of the cone-beam geometry defined as
            (source to detector distance)/(source to center-of-rotation distance).
        weights (float, ndarray): Weights used in mbircone reconstruction, with the same array shape as ``sino``.

        snr_db (float, optional): [Default=40.0] Assumed signal-to-noise ratio of the data in :math:`dB`.
        delta_pixel_image (float, optional): [Default=1.0] Image pixel spacing in :math:`ALU`.
        delta_pixel_detector (float, optional): [Default=1.0] Detector pixel spacing in :math:`ALU`.

    Returns:
        (float): Automatic value of forward model regularization parameter ``sigma_y``.
    """

    # Compute indicator function for sinogram support
    sino_indicator = _sino_indicator(sino)

    # Compute RMS value of sinogram excluding empty space
    signal_rms = np.average(weights * sino ** 2, None, sino_indicator) ** 0.5

    # Convert snr to relative noise standard deviation
    rel_noise_std = 10 ** (-snr_db / 20)
    # compute the default_pixel_pitch = the detector pixel pitch in the image plane given the magnification
    default_pixel_pitch = delta_pixel_detector / magnification

    # Compute the image pixel pitch relative to the default.
    pixel_pitch_relative_to_default = delta_pixel_image / default_pixel_pitch

    # Compute sigma_y and scale by relative pixel pitch
    sigma_y = rel_noise_std * signal_rms * (pixel_pitch_relative_to_default ** 0.5)

    if sigma_y > 0:
        return sigma_y
    else:
        return 1.0


def auto_sigma_prior(sino, magnification, delta_pixel_detector=1.0, sharpness=0.0):
    """ Compute the automatic prior model regularization parameter for use in MBIR reconstruction.
    
    Args:
        sino (float, ndarray): Sinogram data with either 3D shape (num_views, num_det_rows, num_det_channels)
            or 4D shape (num_time_points, num_views, num_det_rows, num_det_channels).
        magnification (float): Magnification of the cone-beam geometry defined as
            (source to detector distance)/(source to center-of-rotation distance).

        delta_pixel_detector (float, optional): [Default=1.0] Detector pixel spacing in :math:`ALU`.
        sharpness (float, optional): [Default=0.0] Controls level of sharpness.
            ``sharpness=0.0`` is neutral; ``sharpness>0`` increases sharpness; ``sharpness<0`` reduces sharpness.

    Returns:
        (float): Automatic value of prior model regularization parameter.
    """
    
    num_det_channels = sino.shape[-1]

    # Compute indicator function for sinogram support
    sino_indicator = _sino_indicator(sino)

    # Compute a typical image value by dividing average sinogram value by a typical projection path length
    typical_img_value = np.average(sino, weights=sino_indicator) / (num_det_channels * delta_pixel_detector / magnification)

    # Compute sigma_x as a fraction of the typical image value
    sigma_prior = (2 ** sharpness) * typical_img_value
    return sigma_prior


def auto_sigma_x(sino, magnification, delta_pixel_detector=1.0, sharpness=0.0):
    """ Compute the automatic value of ``sigma_x`` for use in MBIR reconstruction with qGGMRF prior.

    Args:
        sino (float, ndarray): Sinogram data with either 3D shape (num_views, num_det_rows, num_det_channels)
            or 4D shape (num_time_points, num_views, num_det_rows, num_det_channels).
        magnification (float): Magnification of the cone-beam geometry defined as
            (source to detector distance)/(source to center-of-rotation distance).

        delta_pixel_detector (float, optional): [Default=1.0] Detector pixel spacing in :math:`ALU`.
        sharpness (float, optional): [Default=0.0] Controls level of sharpness.
            ``sharpness=0.0`` is neutral; ``sharpness>0`` increases sharpness; ``sharpness<0`` reduces sharpness

    Returns:
        (float): Automatic value of qGGMRF prior model regularization parameter.
    """
    return 0.2 * auto_sigma_prior(sino, magnification, delta_pixel_detector, sharpness)


def auto_sigma_p(sino, magnification, delta_pixel_detector = 1.0, sharpness = 0.0 ):
    """ Compute the automatic value of ``sigma_p`` for use in MBIR reconstruction with proximal map prior.

    Args:
        sino (float, ndarray): Sinogram data with either 3D shape (num_views, num_det_rows, num_det_channels)
            or 4D shape (num_time_points, num_views, num_det_rows, num_det_channels).
        magnification (float): Magnification of the cone-beam geometry defined as
            (source to detector distance)/(source to center-of-rotation distance).

        delta_pixel_detector (float, optional): [Default=1.0] Detector pixel spacing in :math:`ALU`.
        sharpness (float, optional): [Default=0.0] Controls level of sharpness.
            ``sharpness=0.0`` is neutral; ``sharpness>0`` increases sharpness; ``sharpness<0`` reduces sharpness

    Returns:
        (float): Automatic value of proximal map prior model regularization parameter.
    """
    
    return 2.0 * auto_sigma_prior(sino, magnification, delta_pixel_detector, sharpness)


def auto_image_size(num_det_rows, num_det_channels,
                    delta_det_channel, delta_det_row, delta_pixel_image, magnification):
    """ Compute the automatic image array size for use in MBIR reconstruction.
    
    Args:
        num_det_rows (int): Number of rows in sinogram data.
        num_det_channels (int): Number of channels in sinogram data.
        delta_det_channel (float, optional): [Default=1.0] Detector channel spacing in :math:`ALU`.
        delta_det_row (float, optional): [Default=1.0] Detector row spacing in :math:`ALU`.
        delta_pixel_image (float): Image pixel spacing in :math:`ALU`.
        magnification (float): Magnification of the cone-beam geometry defined as
            (source to detector distance)/(source to center-of-rotation distance).
    
    Returns:
        (int, 3-tuple): Default values for ``num_image_rows``, ``num_image_cols``, ``num_image_slices`` for the
        inputted image measurements.
        
    """
    
    num_image_rows = int(np.round(num_det_channels*((delta_det_channel/delta_pixel_image)/magnification)))
    num_image_cols = num_image_rows
    num_image_slices = int(np.round(num_det_rows*((delta_det_row/delta_pixel_image)/magnification)))
    
    return (num_image_rows, num_image_cols, num_image_slices)


def create_image_params_dict(num_image_rows, num_image_cols, num_image_slices, delta_pixel_image=1.0, image_slice_offset=0.0):
    """ Allocate image parameters as required by ``cone3D.recon`` and ``cone3D.project``.
    
    Args:
        num_image_rows (int): Number of rows in image region.
        num_image_cols (int): Number of columns in image region.
        num_image_slices (int): Number of slices in image region.

        delta_pixel_image (float, optional): [Default=1.0] Image pixel spacing in :math:`ALU`.
        image_slice_offset (float, optional): [Default=0.0] Vertical offset of the image in units of :math:`ALU`.
    
    Returns:
        (dict): Parameters specifying the location and dimensions of a 3D density image.

    """

    imgparams = dict()
    imgparams['N_x'] = num_image_rows
    imgparams['N_y'] = num_image_cols
    imgparams['N_z'] = num_image_slices

    imgparams['Delta_xy'] = delta_pixel_image
    imgparams['Delta_z'] = delta_pixel_image

    imgparams['x_0'] = -imgparams['N_x']*imgparams['Delta_xy']/2.0
    imgparams['y_0'] = -imgparams['N_y']*imgparams['Delta_xy']/2.0
    imgparams['z_0'] = -imgparams['N_z']*imgparams['Delta_z']/2.0 - image_slice_offset
        
    # Deprecated parameters
        
    imgparams['j_xstart_roi'] = -1
    imgparams['j_ystart_roi'] = -1

    imgparams['j_xstop_roi'] = -1
    imgparams['j_ystop_roi'] = -1

    imgparams['j_zstart_roi'] = -1
    imgparams['j_zstop_roi'] = -1

    imgparams['N_x_roi'] = -1
    imgparams['N_y_roi'] = -1
    imgparams['N_z_roi'] = -1

    return imgparams


def create_sino_params_dict(dist_source_detector, magnification,
                        num_views, num_det_rows, num_det_channels,
                        det_channel_offset=0.0, det_row_offset=0.0, rotation_offset=0.0,
                        delta_det_channel=1.0, delta_det_row=1.0):
    """ Allocate sinogram parameters as required by ``cone3D.recon`` and ``cone3D.project``.
    
    Args:
        dist_source_detector (float): Distance between the X-ray source and the detector in units of :math:`ALU`.
        magnification (float): Magnification of the cone-beam geometry defined as
            (source to detector distance)/(source to center-of-rotation distance).
        
        num_views (int): Number of views in sinogram data
        num_det_rows (int): Number of rows in sinogram data
        num_det_channels (int): Number of channels in sinogram data

        det_channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector
            to the source-detector line along a row.
        det_row_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector
            to the source-detector line along a column.
        rotation_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from source-detector line
            to axis of rotation in the object space.
            This is normally set to zero.
        delta_det_channel (float, optional): [Default=1.0] Detector channel spacing in :math:`ALU`.
        delta_det_row (float, optional): [Default=1.0] Detector row spacing in :math:`ALU`.
    
    Returns:
        (dict): Parameters specifying the location and dimensions of the X-ray source and detector.
    """

    sinoparams = dict()
    sinoparams['N_dv'] = num_det_channels
    sinoparams['N_dw'] = num_det_rows
    sinoparams['N_beta'] = num_views
    sinoparams['Delta_dv'] = delta_det_channel
    sinoparams['Delta_dw'] = delta_det_row
    sinoparams['u_s'] = - dist_source_detector / magnification
    sinoparams['u_r'] = 0
    sinoparams['v_r'] = rotation_offset
    sinoparams['u_d0'] = dist_source_detector - dist_source_detector / magnification

    dist_dv_to_detector_corner_from_detector_center = - sinoparams['N_dv'] * sinoparams['Delta_dv'] / 2
    dist_dw_to_detector_corner_from_detector_center = - sinoparams['N_dw'] * sinoparams['Delta_dw'] / 2

    dist_dv_to_detector_center_from_source_detector_line = - det_channel_offset
    dist_dw_to_detector_center_from_source_detector_line = - det_row_offset

    # Corner of detector from source-detector-line
    sinoparams[
        'v_d0'] = dist_dv_to_detector_corner_from_detector_center + dist_dv_to_detector_center_from_source_detector_line
    sinoparams[
        'w_d0'] = dist_dw_to_detector_corner_from_detector_center + dist_dw_to_detector_center_from_source_detector_line

    sinoparams['weightScaler_value'] = -1

    return sinoparams


def recon(sino, angles, dist_source_detector, magnification,
          weights=None, weight_type='unweighted', init_image=0.0, prox_image=None,
          num_image_rows=None, num_image_cols=None, num_image_slices=None,
          delta_det_channel=1.0, delta_det_row=1.0, delta_pixel_image=None,
          det_channel_offset=0.0, det_row_offset=0.0, rotation_offset=0.0, image_slice_offset=0.0,
          sigma_y=None, snr_db=40.0, sigma_x=None, sigma_p=None, p=1.2, q=2.0, T=1.0, num_neighbors=6,
          sharpness=0.0, positivity=True, max_resolutions=None, stop_threshold=0.02, max_iterations=100,
          NHICD=False, num_threads=None, verbose=1, lib_path=__lib_path):
    """ Compute 3D cone beam MBIR reconstruction
    
    Args:
        sino (float, ndarray): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
        angles (float, ndarray): 1D array of view angles in radians.
        dist_source_detector (float): Distance between the X-ray source and the detector in units of :math:`ALU`.
        magnification (float): Magnification of the cone-beam geometry defined as
            (source to detector distance)/(source to center-of-rotation distance).
        
        weights (float, ndarray, optional): [Default=None] 3D weights array with same shape as ``sino``.
            If ``weights`` is not supplied, then ``cone3D.calc_weights`` is used to set weights using ``weight_type``.
        weight_type (string, optional): [Default='unweighted'] Type of noise model used for data.

                - ``'unweighted'`` corresponds to unweighted reconstruction;
                - ``'transmission'`` is the correct weighting for transmission CT with constant dosage;
                - ``'transmission_root'`` is commonly used with transmission CT data to improve image homogeneity;
                - ``'emission'`` is appropriate for emission CT data.
        init_image (float, ndarray, optional): [Default=0.0] Initial value of reconstruction image, specified by either
            a scalar value or a 3D numpy array with shape (num_img_slices, num_img_rows, num_img_cols).
        prox_image (float, ndarray, optional): [Default=None] 3D proximal map input image with shape
            (num_img_slices, num_img_rows, num_img_cols).
        
        num_image_rows (int, optional): [Default=None] Number of rows in reconstructed image.
            If None, automatically set by ``cone3D.auto_image_size``.
        num_image_cols (int, optional): [Default=None] Number of columns in reconstructed image.
            If None, automatically set by ``cone3D.auto_image_size``.
        num_image_slices (int, optional): [Default=None] Number of slices in reconstructed image.
            If None, automatically set by ``cone3D.auto_image_size``.

        delta_det_channel (float, optional): [Default=1.0] Detector channel spacing in :math:`ALU`.
        delta_det_row (float, optional): [Default=1.0] Detector row spacing in :math:`ALU`.
        delta_pixel_image (float, optional): [Default=None] Image pixel spacing in :math:`ALU`.
            If None, automatically set to ``delta_pixel_detector/magnification``.
        
        det_channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector
            to the source-detector line along a row.
        det_row_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector
            to the source-detector line along a column.
        rotation_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from source-detector line
            to axis of rotation in the object space.
            This is normally set to zero.
        image_slice_offset (float, optional): [Default=0.0] Vertical offset of the image in units of :math:`ALU`.
        
        sigma_y (float, optional): [Default=None] Forward model regularization parameter.
            If None, automatically set with ``cone3D.auto_sigma_y``.
        snr_db (float, optional): [Default=40.0] Assumed signal-to-noise ratio of the data in :math:`dB`.
            Ignored if ``sigma_y`` is not None.
        sigma_x (float, optional): [Default=None] qGGMRF prior model regularization parameter.
            If None, automatically set with ``cone3D.auto_sigma_x`` as a function of ``sharpness``.
            If ``prox_image`` is given, ``sigma_p`` is used instead of ``sigma_x`` in the reconstruction.
        sigma_p (float, optional): [Default=None] Proximal map regularization parameter.
            If None, automatically set with ``cone3D.auto_sigma_p`` as a function of ``sharpness``.
            Ignored if ``prox_image`` is None.
        p (float, optional): [Default=1.2] Scalar value in range :math:`[1,2]` that specifies qGGMRF shape parameter.
        q (float, optional): [Default=2.0] Scalar value in range :math:`[p,1]` that specifies qGGMRF shape parameter.
        T (float, optional): [Default=1.0] Scalar value :math:`>0` that specifies the qGGMRF threshold parameter.
        num_neighbors (int, optional): [Default=6] Possible values are :math:`{26,18,6}`.
            Number of neighbors in the qGGMRF neighborhood. More neighbors results in a better
            regularization but a slower reconstruction.
        
        sharpness (float, optional): [Default=0.0] Sharpness of reconstruction.
            ``sharpness=0.0`` is neutral; ``sharpness>0`` increases sharpness; ``sharpness<0`` reduces sharpness.
            Used to calculate ``sigma_x`` and ``sigma_p``.
            Ignored if ``sigma_x`` is not None in qGGMRF mode, or if ``sigma_p`` is not None in proximal map mode.
        positivity (bool, optional): [Default=True] Determines if positivity constraint will be enforced.
        max_resolutions (int, optional): [Default=None] Integer :math:`\geq 0` that specifies the maximum number of grid
            resolutions used to solve MBIR reconstruction problem.
            If None, automatically set by ``cone3D.auto_max_resolutions``.
        stop_threshold (float, optional): [Default=0.02] Relative update stopping threshold, in percent, where relative
            update is given by (average value change) / (average voxel value).
        max_iterations (int, optional): [Default=100] Maximum number of iterations before stopping.

        NHICD (bool, optional): [Default=False] If True, uses non-homogeneous ICD updates.
        num_threads (int, optional): [Default=None] Number of compute threads requested when executed.
            If None, this is set to the number of cores in the system.
        verbose (int, optional): [Default=1] Possible values are {0,1,2}, where 0 is quiet, 1 prints minimal
            reconstruction progress information, and 2 prints the full information.
        lib_path (str, optional): [Default=~/.cache/mbircone] Path to directory containing library of
            forward projection matrices.


    Returns:
        (float, ndarray): 3D reconstruction image with shape (num_img_slices, num_img_rows, num_img_cols) in units of
        :math:`ALU^{-1}`.
    """

    # Internally set
    # NHICD_ThresholdAllVoxels_ErrorPercent=80, NHICD_percentage=15, NHICD_random=20, 
    # zipLineMode=2, N_G=2, numVoxelsPerZiplineMax=200

    if num_threads is None:
        num_threads = cpu_count(logical=False)

    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['OMP_DYNAMIC'] = 'true'
    
    # Set automatic value of max_resolutions
    if max_resolutions is None :
        max_resolutions = auto_max_resolutions(init_image)
    print('max_resolution = ', max_resolutions)

    (num_views, num_det_rows, num_det_channels) = sino.shape

    if delta_pixel_image is None:
        delta_pixel_image = delta_det_channel/magnification
    if num_image_rows is None:
        num_image_rows, _, _ = auto_image_size(num_det_rows, num_det_channels, delta_det_channel, delta_pixel_image,
                                             magnification)
    if num_image_cols is None:
        _, num_image_cols, _ = auto_image_size(num_det_rows, num_det_channels, delta_det_channel, delta_pixel_image,
                                             magnification)
    if num_image_slices is None:
        _, _, num_image_slices = auto_image_size(num_det_rows, num_det_channels, delta_det_channel, delta_pixel_image,
                                               magnification)
    
    sinoparams = create_sino_params_dict(dist_source_detector, magnification,
                                     num_views=num_views, num_det_rows=num_det_rows, num_det_channels=num_det_channels,
                                     det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                     rotation_offset=rotation_offset,
                                     delta_det_channel=delta_det_channel, delta_det_row=delta_det_row)
    
    imgparams = create_image_params_dict(num_image_rows, num_image_cols, num_image_slices,
                                         delta_pixel_image=delta_pixel_image, image_slice_offset=image_slice_offset)
    
    # Make sure that weights do not contain negative entries
    # If weights is provided, and negative entry exists, then do not use the provided weights
    if not ((weights is None) or (np.amin(weights) >= 0.0)):
        warnings.warn("Parameter weights contains negative values; Setting weights = None.")
        weights = None
    # Set automatic values for weights
    if weights is None:
        weights = calc_weights(sino, weight_type)

    # Set automatic value of sigma_y
    if sigma_y is None:
        sigma_y = auto_sigma_y(sino, magnification, weights, snr_db, 
                               delta_pixel_image=delta_pixel_image,
                               delta_pixel_detector=delta_det_channel)

    # Set automatic value of sigma_x
    if sigma_x is None:
        sigma_x = auto_sigma_x(sino, magnification, delta_pixel_detector=delta_det_channel, sharpness=sharpness)

    reconparams = dict()
    reconparams['is_positivity_constraint'] = bool(positivity)
    reconparams['q'] = q
    reconparams['p'] = p
    reconparams['T'] = T
    reconparams['sigmaX'] = sigma_x

    if num_neighbors not in [6, 18, 26]:
        num_neighbors = 6

    if num_neighbors == 6:
        reconparams['bFace'] = 1.0
        reconparams['bEdge'] = -1
        reconparams['bVertex'] = -1

    if num_neighbors == 18:
        reconparams['bFace'] = 1.0
        reconparams['bEdge'] = 0.70710678118
        reconparams['bVertex'] = -1

    if num_neighbors == 26:
        reconparams['bFace'] = 1.0
        reconparams['bEdge'] = 0.70710678118
        reconparams['bVertex'] = 0.57735026919

    reconparams['stopThresholdChange_pct'] = stop_threshold
    reconparams['MaxIterations'] = max_iterations

    reconparams['weightScaler_value'] = sigma_y ** 2

    reconparams['verbosity'] = verbose

    ################ Internally set

    # Weight scalar
    reconparams['weightScaler_domain'] = 'spatiallyInvariant'
    reconparams['weightScaler_estimateMode'] = 'None'

    # Stopping
    reconparams['stopThesholdRWFE_pct'] = 0
    reconparams['stopThesholdRUFE_pct'] = 0
    reconparams['relativeChangeMode'] = 'meanImage'
    reconparams['relativeChangeScaler'] = 0.1
    reconparams['relativeChangePercentile'] = 99.9

    # Zipline
    reconparams['zipLineMode'] = 2
    reconparams['N_G'] = 2
    reconparams['numVoxelsPerZiplineMax'] = 200
    reconparams['numVoxelsPerZipline'] = 200
    reconparams['numZiplines'] = 4

    # NHICD
    reconparams['NHICD_ThresholdAllVoxels_ErrorPercent'] = 80
    reconparams['NHICD_percentage'] = 15
    reconparams['NHICD_random'] = 20
    reconparams['isComputeCost'] = 1
    if NHICD:
        reconparams['NHICD_Mode'] = 'percentile+random'
    else:
        reconparams['NHICD_Mode'] = 'off'

    if prox_image is None:
        reconparams['prox_mode'] = False
        reconparams['sigma_lambda'] = 1
    else:
        reconparams['prox_mode'] = True
        if sigma_p is None:
            sigma_p = auto_sigma_p(sino, magnification, delta_det_channel, sharpness)
        reconparams['sigma_lambda'] = sigma_p

    x = ci.recon_cy(sino, angles, weights, init_image, prox_image,
                    sinoparams, imgparams, reconparams, max_resolutions,
                    num_threads, lib_path)
    return x


def project(image, angles,
            num_det_rows, num_det_channels,
            dist_source_detector, magnification,
            delta_det_channel=1.0, delta_det_row=1.0, delta_pixel_image=None,
            det_channel_offset=0.0, det_row_offset=0.0, rotation_offset=0.0, image_slice_offset=0.0,
            num_threads=None, verbose=1, lib_path=__lib_path):
    """ Compute 3D cone beam forward projection.
    
    Args:
        image (float, ndarray): 3D image to be projected, with shape (num_img_slices, num_img_rows, num_img_cols).
        angles (float, ndarray): 1D array of view angles in radians.

        num_det_rows (int): Number of rows in sinogram data.
        num_det_channels (int): Number of channels in sinogram data.

        dist_source_detector (float): Distance between the X-ray source and the detector in units of :math:`ALU`.
        magnification (float): Magnification of the cone-beam geometry defined as
            (source to detector distance)/(source to center-of-rotation distance).

        delta_det_channel (float, optional): [Default=1.0] Detector channel spacing in :math:`ALU`.
        delta_det_row (float, optional): [Default=1.0] Detector row spacing in :math:`ALU`.
        delta_pixel_image (float, optional): [Default=None] Image pixel spacing in :math:`ALU`.
            If None, automatically set to ``delta_pixel_detector/magnification``.

        det_channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector
            to the source-detector line along a row.
        det_row_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector
            to the source-detector line along a column.
        rotation_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from source-detector line
            to axis of rotation in the object space.
            This is normally set to zero.
        image_slice_offset (float, optional): [Default=0.0] Vertical offset of the image in units of :math:`ALU`.

        num_threads (int, optional): [Default=None] Number of compute threads requested when executed.
            If None, ``num_threads`` is set to the number of cores in the system.
        verbose (int, optional): [Default=1] Possible values are {0,1,2}, where 0 is quiet, 1 prints minimal
            reconstruction progress information, and 2 prints the full information.
        lib_path (str, optional): [Default=~/.cache/mbircone] Path to directory containing library of
            forward projection matrices.

    Returns:
        (float, ndarray): 3D sinogram with shape (num_views, num_det_rows, num_det_channels).
    """

    if num_threads is None:
        num_threads = cpu_count(logical=False)

    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['OMP_DYNAMIC'] = 'true'

    if delta_pixel_image is None:
        delta_pixel_image = delta_det_channel / magnification

    num_views = len(angles)

    sinoparams = create_sino_params_dict(dist_source_detector, magnification,
                                         num_views=num_views, num_det_rows=num_det_rows, num_det_channels=num_det_channels,
                                         det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                         rotation_offset=rotation_offset,
                                         delta_det_channel=delta_det_channel, delta_det_row=delta_det_row)
     
    (num_image_slices, num_image_rows, num_image_cols) = image.shape
    
    imgparams = create_image_params_dict(num_image_rows, num_image_cols, num_image_slices, delta_pixel_image=delta_pixel_image)

    hash_val = _utils.hash_params(angles, sinoparams, imgparams)
    sysmatrix_fname = _utils._gen_sysmatrix_fname(lib_path=lib_path, sysmatrix_name=hash_val[:__namelen_sysmatrix])

    if os.path.exists(sysmatrix_fname):
        os.utime(sysmatrix_fname)  # Update file modified time
    else:
        sysmatrix_fname_tmp = _utils._gen_sysmatrix_fname_tmp(lib_path=lib_path, sysmatrix_name=hash_val[:__namelen_sysmatrix])
        ci.AmatrixComputeToFile_cy(angles, sinoparams, imgparams, sysmatrix_fname_tmp, verbose=verbose)
        os.rename(sysmatrix_fname_tmp, sysmatrix_fname)

    # Collect settings to pass to C
    settings = dict()
    settings['imgparams'] = imgparams
    settings['sinoparams'] = sinoparams
    settings['sysmatrix_fname'] = sysmatrix_fname
    settings['num_threads'] = num_threads

    proj = ci.project(image, settings)
    return proj
