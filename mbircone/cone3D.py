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
    """Compute a binary function that indicates the region of sinogram support.

    Args:
        sino (ndarray):
            numpy array of sinogram data with either 3D shape (num_views,num_det_rows,num_det_channels) or 4D shape (num_time_points,num_views,num_det_rows,num_det_channels)

    Returns:
        int8: A binary value: =1 within sinogram support; =0 outside sinogram support.
    """
    indicator = np.int8(sino > 0.05 * np.mean(np.fabs(sino)))  # for excluding empty space from average
    return indicator


def _distance_line_to_point(A, B, P):
    """Compute the distance from point P to the line passing through points A and B. (Depreciated method)
    
    Args:
        A (float, 2-tuple): (x,y) coordinate of point A
        B (float, 2-tuple): (x,y) coordinate of point B
        P (float, 2-tuple): (x,y) coordinate of point P
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
    """Compute the weights used in MBIR reconstruction.

    Args:
        sino (ndarray): numpy array of sinogram data with either 3D shape (num_views,num_det_rows,num_det_channels) or 4D shape (num_time_points,num_views,num_det_rows,num_det_channels)
        weight_type (string):[Default=0] Type of noise model used for data.

            If weight_type="unweighted"        => weights = numpy.ones_like(sino)

            If weight_type="transmission"      => weights = numpy.exp(-sino)

            If weight_type="transmission_root" => weights = numpy.exp(-sino/2)

            If weight_type="emission"         => weights = 1/(sino + 0.1)

    Returns:
        ndarray: numpy array of weights with same shape as sino.

    Raises:
        Exception: Description
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


def auto_max_resolutions(init_image):
    """Compute the automatic value of ``max_resolutions`` for use in MBIR reconstruction.

    Args:
        init_image (ndarray): Initial image for reconstruction.
    Returns:
        int: Automatic value of ``max_resolutions``.
    """
    # Default value of max_resolutions
    max_resolutions = 2
    if isinstance(init_image, np.ndarray) and (init_image.ndim == 3):
        # print('Init image present. Setting max_resolutions = 0.')
        max_resolutions = 0

    return max_resolutions


def auto_sigma_y(sino, magnification, weights, snr_db=40.0, delta_pixel_image=1.0, delta_pixel_detector=1.0):
    """Compute the automatic value of ``sigma_y`` for use in MBIR reconstruction.

    Args:
        sino (ndarray): numpy array of sinogram data with either 3D shape (num_views,num_det_rows,num_det_channels) or 4D shape (num_time_points,num_views,num_det_rows,num_det_channels)
        magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
        weights (ndarray):
            numpy array of weights with same shape as sino.
            The parameters weights should be the same values as used in mbircone reconstruction.
        snr_db (float, optional):
            [Default=40.0] Scalar value that controls assumed signal-to-noise ratio of the data in dB.
        delta_pixel_image (float, optional):
            [Default=1.0] Scalar value of pixel spacing in :math:`ALU`.
        delta_pixel_detector (float, optional):
            [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.


    Returns:
        float: Automatic values of regularization parameter.
    """

    # Compute indicator function for sinogram support
    sino_indicator = _sino_indicator(sino)

    # compute RMS value of sinogram excluding empty space
    signal_rms = np.average(weights * sino ** 2, None, sino_indicator) ** 0.5

    # convert snr to relative noise standard deviation
    rel_noise_std = 10 ** (-snr_db / 20)
    # compute the default_pixel_pitch = the detector pixel pitch in the image plane given the magnification
    default_pixel_pitch = delta_pixel_detector / magnification

    # compute the image pixel pitch relative to the default.
    pixel_pitch_relative_to_default = delta_pixel_image / default_pixel_pitch

    # compute sigma_y and scale by relative pixel pitch
    sigma_y = rel_noise_std * signal_rms * (pixel_pitch_relative_to_default ** 0.5)

    if sigma_y > 0:
        return sigma_y
    else:
        return 1.0


def auto_sigma_prior(sino, magnification, delta_pixel_detector=1.0, sharpness=0.0):
    """Compute the automatic value of prior model regularization for use in MBIR reconstruction.
    
    Args:
        sino (ndarray): numpy array of sinogram data with either 3D shape (num_views,num_det_rows,num_det_channels) or 4D shape (num_time_points,num_views,num_det_rows,num_det_channels)
        magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
        delta_pixel_detector (float, optional):
            [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.
        sharpness (float, optional):
            [Default=0.0] Scalar value that controls level of sharpness.
            ``sharpness=0.0`` is neutral; ``sharpness>0`` increases sharpness; ``sharpness<0`` reduces sharpness
    Returns:
        float: Automatic value of regularization parameter.
    """

    num_det_channels = sino.shape[-1]

    # Compute indicator function for sinogram support
    sino_indicator = _sino_indicator(sino)

    # Compute a typical image value by dividing average sinogram value by a typical projection path length
    typical_img_value = np.average(sino, weights=sino_indicator) / (
                num_det_channels * delta_pixel_detector / magnification)

    # Compute sigma_x as a fraction of the typical image value
    sigma_prior = (2 ** sharpness) * typical_img_value
    return sigma_prior


def auto_sigma_x(sino, magnification, delta_pixel_detector=1.0, sharpness=0.0):
    """Compute the automatic value of ``sigma_x`` for use in MBIR reconstruction.

    Args:
        sino (ndarray): numpy array of sinogram data with either 3D shape (num_views,num_det_rows,num_det_channels) or 4D shape (num_time_points,num_views,num_det_rows,num_det_channels)
        magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
        delta_pixel_detector (float, optional):
            [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.
        sharpness (float, optional):
            [Default=0.0] Scalar value that controls level of sharpness.
            ``sharpness=0.0`` is neutral; ``sharpness>0`` increases sharpness; ``sharpness<0`` reduces sharpness

    Returns:
        float: Automatic value of regularization parameter.
    """
    return 0.2 * auto_sigma_prior(sino, magnification, delta_pixel_detector, sharpness)


def auto_sigma_p(sino, magnification, delta_pixel_detector=1.0, sharpness=0.0):
    """Compute the automatic value of ``sigma_p`` for use in proximal map estimation.

    Args:
        sino (ndarray): numpy array of sinogram data with either 3D shape (num_views,num_det_rows,num_det_channels) or 4D shape (num_time_points,num_views,num_det_rows,num_det_channels)
        magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
        delta_pixel_detector (float, optional): [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.
        sharpness (float, optional): [Default=0.0] Scalar value that controls level of sharpness.
            ``sharpness=0.0`` is neutral; ``sharpness>0`` increases sharpness; ``sharpness<0`` reduces sharpness

    Returns:
        float: Automatic value of regularization parameter.
    """

    return 2.0 * auto_sigma_prior(sino, magnification, delta_pixel_detector, sharpness)


def auto_image_size(num_det_rows, num_det_channels, delta_pixel_detector, delta_pixel_image, magnification):
    """Compute the automatic image size for use in recon.
    
    Args:
        num_det_rows (int): Number of rows in sinogram data
        num_det_channels (int): Number of channels in sinogram data
        delta_pixel_detector (float): [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.
        delta_pixel_image (float): [Default=None] Scalar value of image pixel spacing in :math:`ALU`.
            If None, automatically set to delta_pixel_detector/magnification
        magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
    
    Returns:
        (int, 3-tuple): Default values for num_image_rows, num_image_cols, num_image_slices for the inputted image measurements.
        
    """

    num_image_rows = int(np.round(num_det_channels * ((delta_pixel_detector / delta_pixel_image) / magnification)))
    num_image_cols = num_image_rows
    num_image_slices = int(np.round(num_det_rows * ((delta_pixel_detector / delta_pixel_image) / magnification)))

    return num_image_rows, num_image_cols, num_image_slices


def create_sino_params_dict(dist_source_detector, magnification,
                            num_views, num_det_rows, num_det_channels,
                            det_channel_offset=0.0, det_row_offset=0.0, rotation_offset=0.0,
                            delta_pixel_detector=1.0):
    """ Compute sinogram parameters specify coordinates and bounds relating to the sinogram.
        For detailed specifications of sinoparams, see cone3D.interface_cy_c
    
    Args:
        dist_source_detector (float): Distance between the X-ray source and the detector in units of ALU
        magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
        
        num_views (int): Number of views in sinogram data
        num_det_rows (int): Number of rows in sinogram data
        num_det_channels (int): Number of channels in sinogram data

        det_channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a row.
        det_row_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a column.
        rotation_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from source-detector line to axis of rotation in the object space.
            This is normally set to zero.
        delta_pixel_detector (float, optional): [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.
    
    Returns:
        Dictionary containing sino parameters as required by the Cython code
    """

    sinoparams = dict()
    sinoparams['N_dv'] = num_det_channels
    sinoparams['N_dw'] = num_det_rows
    sinoparams['N_beta'] = num_views
    sinoparams['Delta_dv'] = delta_pixel_detector
    sinoparams['Delta_dw'] = delta_pixel_detector
    sinoparams['u_s'] = - dist_source_detector / magnification
    sinoparams['u_r'] = 0
    sinoparams['v_r'] = rotation_offset
    sinoparams['u_d0'] = dist_source_detector - dist_source_detector / magnification

    dist_dv_to_detector_corner_from_detector_center = - sinoparams['N_dv'] * sinoparams['Delta_dv'] / 2
    dist_dw_to_detector_corner_from_detector_center = - sinoparams['N_dw'] * sinoparams['Delta_dw'] / 2

    dist_dv_to_detector_center_from_source_detector_line = - det_channel_offset
    dist_dw_to_detector_center_from_source_detector_line = - det_row_offset

    # corner of detector from source-detector-line
    sinoparams[
        'v_d0'] = dist_dv_to_detector_corner_from_detector_center + dist_dv_to_detector_center_from_source_detector_line
    sinoparams[
        'w_d0'] = dist_dw_to_detector_corner_from_detector_center + dist_dw_to_detector_center_from_source_detector_line

    sinoparams['weightScaler_value'] = -1

    return sinoparams


def create_image_params_dict(num_image_rows, num_image_cols, num_image_slices, delta_pixel_image=1.0,
                             image_slice_offset=0.0):
    """ Allocate imageparam parameters as required by certain C methods.
        Can be used to describe a region of projection (i.e., when an image is available in ``project`` method), or to specify a region of reconstruction.
        For detailed specifications of imageparams, see cone3D.interface_cy_c
    
    Args:
        num_image_rows (int): Integer number of rows in image region.
        num_image_cols (int): Integer number of columns in image region.
        num_image_slices (int): Integer number of slices in image region.
        delta_pixel_image (float, optional): [Default=1.0] Scalar value of image pixel spacing in :math:`ALU`.
        image_slice_offset (float, optional): [Default=0.0] Float that controls vertical offset of the center slice for the reconstruction in units of ALU
    
    Returns:
        Dictionary containing sino parameters as required by the Cython code
    """

    imgparams = dict()
    imgparams['N_x'] = num_image_rows
    imgparams['N_y'] = num_image_cols
    imgparams['N_z'] = num_image_slices

    imgparams['Delta_xy'] = delta_pixel_image
    imgparams['Delta_z'] = delta_pixel_image

    imgparams['x_0'] = -imgparams['N_x'] * imgparams['Delta_xy'] / 2.0
    imgparams['y_0'] = -imgparams['N_y'] * imgparams['Delta_xy'] / 2.0
    imgparams['z_0'] = -imgparams['N_z'] * imgparams['Delta_z'] / 2.0 - image_slice_offset

    # depreciated parameters

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


def create_sino_params_dict_lamino(num_views, num_det_rows, num_det_channels,
                                   theta,
                                   det_channel_offset=0.0,
                                   delta_pixel_detector=1.0):
    """ Compute sinogram parameters specify coordinates and bounds relating to the sinogram
        Function signatures are in laminogram coordinates, output corresponds to sinogram.
        All geometry specifications (i.e. function arguments) are given in laminography coordinates.
        For detailed specifications of sinoparams, see cone3D.interface_cy_c
  
    Args:
        num_views (int): Number of views in laminogram data
        num_det_rows (int): Number of rows in laminogram data
        num_det_channels (int): Number of channels in laminogram data

        theta (float): Laminographic angle; pi/2 - grazing angle

        det_channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the "projected axis of rotation" along a row.
        delta_pixel_detector (float, optional): [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.

    Returns:
        Dictionary containing sino parameters as required by the Cython code
    """

    # Place source at a large distance with respect to detector size.
    dist_factor = 500
    dist_source_detector = dist_factor * delta_pixel_detector * max(num_det_channels, num_det_rows) ** 2

    sinoparams = dict()
    sinoparams['N_dv'] = num_det_channels
    sinoparams['N_dw'] = num_det_rows
    sinoparams['N_beta'] = num_views
    sinoparams['Delta_dv'] = delta_pixel_detector
    sinoparams['Delta_dw'] = delta_pixel_detector / math.sin(theta)
    sinoparams['u_s'] = - dist_source_detector
    sinoparams['u_r'] = 0
    sinoparams['v_r'] = 0
    sinoparams['u_d0'] = 0

    dist_dv_to_detector_corner_from_detector_center = - sinoparams['N_dv'] * sinoparams['Delta_dv'] / 2
    dist_dw_to_detector_corner_from_detector_center = - sinoparams['N_dw'] * sinoparams['Delta_dw'] / 2

    # Real geometry effect
    dist_dv_to_detector_center_from_source_detector_line = - det_channel_offset
    # Set artificially to create angle
    dist_dw_to_detector_center_from_source_detector_line = - dist_source_detector / math.tan(theta)

    # Corner of detector from source-detector-line
    sinoparams[
        'v_d0'] = dist_dv_to_detector_corner_from_detector_center + dist_dv_to_detector_center_from_source_detector_line
    sinoparams[
        'w_d0'] = dist_dw_to_detector_corner_from_detector_center + dist_dw_to_detector_center_from_source_detector_line

    sinoparams['weightScaler_value'] = -1

    return sinoparams


def create_image_params_dict_lamino(sinoparams, num_image_rows, num_image_cols, num_image_slices, theta,
                                  delta_pixel_image=1.0, image_slice_offset=0.0):
    """ Allocate imageparams parameters as required by certain C methods.
        Can be used to describe a region of projection (i.e., when an image is available in ``project`` method), or to specify a region of reconstruction.
        All geometry specifications (i.e. function arguments) are given in laminography coordinates.
        For detailed specifications of imageparams, see cone3D.interface_cy_c
    
    Args:
        sinoparams (dict): Dictionary containing sinogram parameters as required by the Cython code
        num_image_rows (int): Integer number of rows in image region.
        num_image_cols (int): Integer number of columns in image region.
        num_image_slices (int): Integer number of slices in image region.
        theta (float): Laminographic angle, equal to pi/2 - grazing angle
        delta_pixel_image (float, optional): [Default=None] Scalar value of image pixel spacing in :math:`ALU`.
            If None, automatically set to delta_pixel_detector/magnification
        image_slice_offset (float, optional): [Default=0.0] Float that controls vertical offset of the center slice for the reconstruction in units of ALU

        Returns:
            Dictionary containing image parameters as required by the Cython code
    """

    dist_source_detector = -sinoparams['u_s']

    imgparams = dict()
    imgparams['N_x'] = num_image_rows
    imgparams['N_y'] = num_image_cols
    imgparams['N_z'] = num_image_slices

    imgparams['Delta_xy'] = delta_pixel_image
    imgparams['Delta_z'] = delta_pixel_image

    imgparams['x_0'] = -imgparams['N_x'] * imgparams['Delta_xy'] / 2.0
    imgparams['y_0'] = -imgparams['N_y'] * imgparams['Delta_xy'] / 2.0
    imgparams['z_0'] = -imgparams['N_z'] * imgparams['Delta_z'] / 2.0 - image_slice_offset - (dist_source_detector / math.tan(theta))

    # Find coordinates of roi within the voxel image
    imgparams['j_xstart_roi'] = -1
    imgparams['j_ystart_roi'] = -1
    imgparams['j_zstart_roi'] = -1

    imgparams['j_xstop_roi'] = -1
    imgparams['j_ystop_roi'] = -1
    imgparams['j_zstop_roi'] = -1

    imgparams['N_x_roi'] = -1
    imgparams['N_y_roi'] = -1
    imgparams['N_z_roi'] = -1

    return imgparams


def recon(sino, angles, dist_source_detector, magnification,
          geometry='cone',
          weights=None, weight_type='unweighted', init_image=0.0, prox_image=None,
          num_image_rows=None, num_image_cols=None, num_image_slices=None,
          delta_pixel_detector=1.0, delta_pixel_image=None,
          det_channel_offset=0.0, det_row_offset=0.0, rotation_offset=0.0, image_slice_offset=0.0,
          theta=math.pi / 2,
          sigma_y=None, snr_db=40.0, sigma_x=None, sigma_p=None, p=1.2, q=2.0, T=1.0, num_neighbors=6,
          sharpness=0.0, positivity=True, max_resolutions=None, stop_threshold=0.02, max_iterations=100,
          num_threads=None, NHICD=False, lib_path=__lib_path,
          verbose=1):
    """Compute 3D cone beam MBIR reconstruction
    
    Args:
        sino (ndarray): 3D sinogram array with shape (num_views, num_det_rows, num_det_channels)
        angles (ndarray): 1D view angles array in radians.
        dist_source_detector (float): Distance between the X-ray source and the detector in units of ALU
        magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
        
        geometry (string, optional): [Default='cone'] This can be 'cone' or 'lamino'.
            If geometry=='cone', runs a standard cone-beam reconstruction.
            If geometry=='lamino', runs a parallel-beam laminography reconstruction and ignores parameters dist_source_detector, magnification, row_offset, rotation_offset.
        
        weights (ndarray, optional): [Default=None] 3D weights array with same shape as sino.
        weight_type (string, optional): [Default='unweighted'] Type of noise model used for data.
            If the ``weights`` array is not supplied, then the function ``cone3D.calc_weights`` is used to set weights using specified ``weight_type`` parameter.
            
                - Option "unweighted" corresponds to unweighted reconstruction;
                - Option "transmission" is the correct weighting for transmission CT with constant dosage;
                - Option "transmission_root" is commonly used with transmission CT data to improve image homogeneity;
                - Option "emission" is appropriate for emission CT data.
        init_image (ndarray, optional): [Default=0.0] Initial value of reconstruction image, specified by either a scalar value or a 3D numpy array with shape (num_img_slices,num_img_rows,num_img_cols)
        prox_image (ndarray, optional): [Default=None] 3D proximal map input image. 3D numpy array with shape (num_img_slices,num_img_rows,num_img_cols)
        
        num_image_rows (int, optional): [Default=None] Integer number of rows in reconstructed image.
            If None, automatically set by auto_image_size.
        num_image_cols (int, optional): [Default=None] Integer number of columns in reconstructed image.
            If None, automatically set by auto_image_size.
        num_image_slices (int, optional): [Default=None] Integer number of slices in reconstructed image.
            If None, automatically set by auto_image_size.

        delta_pixel_detector (float, optional): [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.
        delta_pixel_image (float, optional): [Default=None] Scalar value of image pixel spacing in :math:`ALU`.
            If None, automatically set to delta_pixel_detector/magnification
        
        det_channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a row.
        det_row_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a column.
        rotation_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from source-detector line to axis of rotation in the object space.
            This is normally set to zero.
        image_slice_offset (float, optional): [Default=0.0] Float that controls vertical offset of the center slice for the reconstruction in units of ALU

        theta (float, optional): [Default=math.pi / 2] Laminographic angle; pi/2 - grazing angle

        sigma_y (float, optional): [Default=None] Scalar value of noise standard deviation parameter.
            If None, automatically set with auto_sigma_y.
        snr_db (float, optional): [Default=40.0] Scalar value that controls assumed signal-to-noise ratio of the data in dB.
            Ignored if sigma_y is not None.
        sigma_x (float, optional): [Default=None] Scalar value :math:`>0` that specifies the qGGMRF scale parameter.
            Ignored if prox_image is not None.
            If None and prox_image is also None, automatically set with auto_sigma_x. Regularization should be controled with the ``sharpness`` parameter, but ``sigma_x`` can be set directly by expert users.
        sigma_p (float, optional): [Default=None] Scalar value :math:`>0` that specifies the proximal map parameter.
            Ignored if prox_image is None.
            If None and proximal image is not None, automatically set with auto_sigma_p. Regularization should be controled with the ``sharpness`` parameter, but ``sigma_p`` can be set directly by expert users.
        p (float, optional): [Default=1.2] Scalar value in range :math:`[1,2]` that specifies the qGGMRF shape parameter.
        q (float, optional): [Default=2.0] Scalar value in range :math:`[p,1]` that specifies the qGGMRF shape parameter.
        T (float, optional): [Default=1.0] Scalar value :math:`>0` that specifies the qGGMRF threshold parameter.
        num_neighbors (int, optional): [Default=6] Possible values are {26,18,6}.
            Number of neightbors in the qggmrf neighborhood. Higher number of neighbors result in a better regularization but a slower reconstruction.
        
        sharpness (float, optional): [Default=0.0]
            Scalar value that controls level of sharpness in the reconstruction
            ``sharpness=0.0`` is neutral; ``sharpness>0`` increases sharpness; ``sharpness<0`` reduces sharpness.
            Ignored if ``sigma_x`` is not None in qGGMRF mode, or if ``sigma_p`` is not None in proximal map mode.
        positivity (bool, optional): [Default=True] Boolean value that determines if positivity constraint is enforced.
            The positivity parameter defaults to True; however, it should be changed to False when used in applications that can generate negative image values.
        max_resolutions (int, optional): [Default=None] Integer >=0 that specifies the maximum number of grid
            resolutions used to solve MBIR reconstruction problem.
            If None, automatically set with auto_max_resolutions to 0 if inital image is provided and 2 otherwise.
        stop_threshold (float, optional): [Default=0.02] Scalar valued stopping threshold in percent.
            If stop_threshold=0.0, then run max iterations.
        max_iterations (int, optional): [Default=100] Integer valued specifying the maximum number of iterations.
        num_threads (int, optional): [Default=None] Number of compute threads requested when executed.
            If None, num_threads is set to the number of cores in the system
        NHICD (bool, optional): [Default=False] If true, uses Non-homogeneous ICD updates
        lib_path (str, optional): [Default=~/.cache/mbircone] Path to directory containing library of forward projection matrices.
        verbose (int, optional): [Default=1] Possible values are {0,1,2}, where 0 is quiet, 1 prints minimal reconstruction progress information, and 2 prints the full information.
        
    Returns:
        3D numpy array: 3D reconstruction with shape (num_img_slices, num_img_rows, num_img_cols) in units of :math:`ALU^{-1}`.
    """

    # Internally set
    # NHICD_ThresholdAllVoxels_ErrorPercent=80, NHICD_percentage=15, NHICD_random=20, 
    # zipLineMode=2, N_G=2, numVoxelsPerZiplineMax=200

    if num_threads is None:
        num_threads = cpu_count(logical=False)

    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['OMP_DYNAMIC'] = 'true'

    # Set automatic value of max_resolutions
    if max_resolutions is None:
        max_resolutions = auto_max_resolutions(init_image)
    print('max_resolution = ', max_resolutions)

    (num_views, num_det_rows, num_det_channels) = sino.shape

    if geometry == 'lamino':
        magnification = 1
    if delta_pixel_image is None:
        delta_pixel_image = delta_pixel_detector / magnification

    if num_image_rows is None:
        num_image_rows, _, _ = auto_image_size(num_det_rows, num_det_channels, delta_pixel_detector, delta_pixel_image,
                                               magnification)
    if num_image_cols is None:
        _, num_image_cols, _ = auto_image_size(num_det_rows, num_det_channels, delta_pixel_detector, delta_pixel_image,
                                               magnification)
    if num_image_slices is None:
        _, _, num_image_slices = auto_image_size(num_det_rows, num_det_channels, delta_pixel_detector,
                                                 delta_pixel_image, magnification)

    if geometry == 'cone':
        sinoparams = create_sino_params_dict(dist_source_detector, magnification,
                                             num_views=num_views, num_det_rows=num_det_rows,
                                             num_det_channels=num_det_channels,
                                             det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                             rotation_offset=rotation_offset,
                                             delta_pixel_detector=delta_pixel_detector)

        imgparams = create_image_params_dict(num_image_rows, num_image_cols, num_image_slices,
                                             delta_pixel_image=delta_pixel_image, image_slice_offset=image_slice_offset)
    elif geometry == 'lamino':
        sinoparams = create_sino_params_dict_lamino(num_views, num_det_rows, num_det_channels,
                                                    theta,
                                                    det_channel_offset=det_channel_offset,
                                                    delta_pixel_detector=delta_pixel_detector)

        imgparams = create_image_params_dict_lamino(sinoparams, num_image_rows, num_image_cols, num_image_slices, theta,
                                                  delta_pixel_image=delta_pixel_image,
                                                  image_slice_offset=image_slice_offset)
    else:
        raise Exception("geometry: undefined geometry {}".format(geometry))

    # make sure that weights do not contain negative entries
    # if weights is provided, and negative entry exists, then do not use the provided weights
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
                               delta_pixel_detector=delta_pixel_detector)

    # Set automatic value of sigma_x
    if sigma_x is None:
        sigma_x = auto_sigma_x(sino, magnification, delta_pixel_detector=delta_pixel_detector, sharpness=sharpness)

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
            sigma_p = auto_sigma_p(sino, magnification, delta_pixel_detector, sharpness)
        reconparams['sigma_lambda'] = sigma_p

    x = ci.recon_cy(sino, angles, weights, init_image, prox_image,
                    sinoparams, imgparams, reconparams, max_resolutions,
                    num_threads, lib_path)
    return x


def project(image, angles,
            num_det_rows, num_det_channels,
            dist_source_detector, magnification,
            geometry='cone',
            det_channel_offset=0.0, det_row_offset=0.0, rotation_offset=0.0,
            delta_pixel_detector=1.0, delta_pixel_image=None,
            theta=math.pi / 2,
            num_threads=None, verbose=1, lib_path=__lib_path):
    """Compute 3D cone beam forward-projection.
    
    Args:
        image (ndarray):
            3D numpy array of image being forward projected.
            The image is a 3D array with a shape of (num_img_slices, num_img_rows, num_img_cols)
        angles (ndarray): 1D view angles array in radians.

        num_det_rows (int): Number of rows in sinogram data
        num_det_channels (int): Number of channels in sinogram data

        dist_source_detector (float): Distance between the X-ray source and the detector in units of ALU
        magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).

        geometry (string, optional): [Default='cone'] This can be 'cone' or 'lamino'.
            If geometry=='cone', runs a standard cone-beam reconstruction.
            If geometry=='lamino', runs a parallel-beam laminography reconstruction and ignores parameters dist_source_detector, magnification, row_offset, rotation_offset.

        det_channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a row.
        det_row_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a column.
        rotation_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from source-detector line to axis of rotation in the object space.
            This is normally set to zero.
        
        delta_pixel_detector (float, optional): [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.
        delta_pixel_image (float, optional): [Default=None] Scalar value of image pixel spacing in :math:`ALU`.
            If None, automatically set to delta_pixel_detector/magnification

        theta (float, optional): [Default=math.pi / 2] Laminographic angle; pi/2 - grazing angle

        num_threads (int, optional): [Default=None] Number of compute threads requested when executed.
            If None, num_threads is set to the number of cores in the system
        verbose (int, optional): [Default=1] Possible values are {0,1,2}, where 0 is quiet, 1 prints minimal reconstruction progress information, and 2 prints the full information.
        lib_path (str, optional): [Default=~/.cache/mbircone] Path to directory containing library of forward projection matrices.
    Returns:
        ndarray: 3D numpy array containing sinogram with shape (num_views, num_det_rows, num_det_channels).
    """

    if num_threads is None:
        num_threads = cpu_count(logical=False)

    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['OMP_DYNAMIC'] = 'true'

    if delta_pixel_image is None:
        delta_pixel_image = delta_pixel_detector / magnification

    num_views = len(angles)

    if geometry == 'cone':
        sinoparams = create_sino_params_dict(dist_source_detector, magnification,
                                             num_views=num_views, num_det_rows=num_det_rows,
                                             num_det_channels=num_det_channels,
                                             det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                             rotation_offset=rotation_offset,
                                             delta_pixel_detector=delta_pixel_detector)

        (num_image_slices, num_image_rows, num_image_cols) = image.shape

        imgparams = create_image_params_dict(num_image_rows, num_image_cols, num_image_slices,
                                             delta_pixel_image=delta_pixel_image)
    elif geometry == 'lamino':
        sinoparams = create_sino_params_dict_lamino(num_views, num_det_rows, num_det_channels,
                                                    theta,
                                                    det_channel_offset=det_channel_offset,
                                                    delta_pixel_detector=delta_pixel_detector)

        (num_image_slices, num_image_rows, num_image_cols) = image.shape

        imgparams = create_image_params_dict_lamino(sinoparams, num_image_rows, num_image_cols, num_image_slices, theta,
                                                  delta_pixel_image=delta_pixel_image)
    else:
        raise Exception("geometry: undefined geometry {}".format(geometry))

    hash_val = _utils.hash_params(angles, sinoparams, imgparams)
    sysmatrix_fname = _utils._gen_sysmatrix_fname(lib_path=lib_path, sysmatrix_name=hash_val[:__namelen_sysmatrix])

    if os.path.exists(sysmatrix_fname):
        os.utime(sysmatrix_fname)  # update file modified time
    else:
        sysmatrix_fname_tmp = _utils._gen_sysmatrix_fname_tmp(lib_path=lib_path,
                                                              sysmatrix_name=hash_val[:__namelen_sysmatrix])
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
