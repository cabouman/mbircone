import math
from psutil import cpu_count
import shutil
import numpy as np
import os
import hashlib
import mbircone.interface_cy_c as ci
import random
import warnings

__lib_path = os.path.join(os.path.expanduser('~'), '.cache', 'mbircone')
__namelen_sysmatrix = 20

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


def _gen_sysmatrix_fname(lib_path=__lib_path, sysmatrix_name='object'):
    os.makedirs(os.path.join(lib_path, 'sysmatrix'), exist_ok=True)

    sysmatrix_fname = os.path.join(lib_path, 'sysmatrix', sysmatrix_name + '.sysmatrix')

    return sysmatrix_fname


def _gen_sysmatrix_fname_tmp(lib_path=__lib_path, sysmatrix_name='object'):
    sysmatrix_fname_tmp = os.path.join(lib_path, 'sysmatrix',
                                       sysmatrix_name + '_pid' + str(os.getpid()) + '_rndnum' + str(
                                           random.randint(0, 1000)) + '.sysmatrix')

    return sysmatrix_fname_tmp


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
    """Compute the distance from point P to the line passing through points A and B
    
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


def hash_params(angles, sinoparams, imgparams):
    hash_input = str(sinoparams) + str(imgparams) + str(np.around(angles, decimals=6))

    hash_val = hashlib.sha512(hash_input.encode()).hexdigest()

    return hash_val


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
    typical_img_value = np.average(sino, weights=sino_indicator) / (num_det_channels * delta_pixel_detector / magnification)

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


def auto_sigma_p(sino, magnification, delta_pixel_detector = 1.0, sharpness = 0.0 ):
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

def compute_sino_params(dist_source_detector, magnification,
                        num_views, num_det_rows, num_det_channels,
                        channel_offset=0.0, row_offset=0.0, rotation_offset=0.0,
                        delta_pixel_detector=1.0):
    """ Compute sinogram parameters specify coordinates and bounds relating to the sinogram
        For detailed specifications of sinoparams, see cone3D.interface_cy_c
    
    Args:
        dist_source_detector (float): Distance between the X-ray source and the detector in units of ALU
        magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
        
        num_views (int): Number of views in sinogram data
        num_det_rows (int): Number of rows in sinogram data
        num_det_channels (int): Number of channels in sinogram data

        channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a row.
        row_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a column.
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

    dist_dv_to_detector_center_from_source_detector_line = - channel_offset
    dist_dw_to_detector_center_from_source_detector_line = - row_offset

    # corner of detector from source-detector-line
    sinoparams[
        'v_d0'] = dist_dv_to_detector_corner_from_detector_center + dist_dv_to_detector_center_from_source_detector_line
    sinoparams[
        'w_d0'] = dist_dw_to_detector_corner_from_detector_center + dist_dw_to_detector_center_from_source_detector_line

    sinoparams['weightScaler_value'] = -1

    return sinoparams


def compute_img_params(sinoparams, delta_pixel_image=None, ror_radius=None):
    """ Compute image parameters that specify coordinates and bounds relating to the image.
        For detailed specifications of imgparams, see cone3D.interface_cy_c
    
    Args:
        sinoparams (dict): Dictionary containing sinogram parameters as required by the Cython code
        delta_pixel_image (float, optional): [Default=None] Scalar value of image pixel spacing in :math:`ALU`.
            If None, automatically set to delta_pixel_detector/magnification
        ror_radius (float, optional): [Default=None] Scalar value of radius of reconstruction in :math:`ALU`.
            If None, automatically set.
            Pixels outside the radius ror_radius in the :math:`(x,y)` plane are disregarded in the reconstruction.
           
    Returns:
        Dictionary containing image parameters as required by the Cython code
     
    """
    
    # port code from https://github.com/cabouman/OpenMBIR-ConeBeam/blob/fbf3eddcadad1bd1cfb657f58c5b32f1204a12d1/utils/Preprocessing/Modular_PreprocessingRoutines/computeImgParams.m

    # Part 1: find radius of circle
    # lower cone point P0
    P0 = (sinoparams['u_d0'], sinoparams['v_d0'])

    # upper cone point P1
    P1 = (sinoparams['u_d0'], sinoparams['v_d0'] + sinoparams['N_dv'] * sinoparams['Delta_dv']);

    # source point S
    S = (sinoparams['u_s'], 0);

    # Rotation center point C
    C = (sinoparams['u_r'], sinoparams['v_r']);

    # r_0 = distance{ line(P0,S), C }
    r_0 = _distance_line_to_point(P0, S, C)

    # r_1 = distance{ line(P1,S), C }
    r_1 = _distance_line_to_point(P1, S, C)

    r = max(r_0, r_1)

    if ror_radius is not None:
        r = ror_radius

    # #### Part 2: assignment of parameters ####

    imgparams = dict()
    imgparams['Delta_xy'] = delta_pixel_image
    imgparams['Delta_z'] = delta_pixel_image

    imgparams['x_0'] = -(r + imgparams['Delta_xy'] / 2)
    imgparams['y_0'] = imgparams['x_0']

    imgparams['N_x'] = 2 * math.ceil(r / imgparams['Delta_xy']) + 1
    imgparams['N_y'] = imgparams['N_x']

    ## Computation of z_0 and N_z

    x_1 = imgparams['x_0'] + imgparams['N_x'] * imgparams['Delta_xy']
    y_1 = x_1

    R_00 = math.sqrt(imgparams['x_0'] ** 2 + imgparams['y_0'] ** 2)
    R_10 = math.sqrt(x_1 ** 2 + imgparams['y_0'] ** 2)
    R_01 = math.sqrt(imgparams['x_0'] ** 2 + y_1 ** 2)
    R_11 = math.sqrt(x_1 ** 2 + y_1 ** 2)

    R = max(R_00, R_10, R_01, R_11)

    w_1 = sinoparams['w_d0'] + sinoparams['N_dw'] * sinoparams['Delta_dw']

    z_0 = min(sinoparams['w_d0'] * (R - sinoparams['u_s']) / (sinoparams['u_d0'] - sinoparams['u_s']),
              sinoparams['w_d0'] * (-R - sinoparams['u_s']) / (sinoparams['u_d0'] - sinoparams['u_s']))

    z_1 = max(w_1 * (R - sinoparams['u_s']) / (sinoparams['u_d0'] - sinoparams['u_s']),
              w_1 * (-R - sinoparams['u_s']) / (sinoparams['u_d0'] - sinoparams['u_s']))

    imgparams['z_0'] = z_0
    imgparams['N_z'] = math.ceil((z_1 - z_0) / (imgparams['Delta_z']))

    ## ROI parameters

    R_roi = min(r_0, r_1) - imgparams['Delta_xy']

    w_0_roi = sinoparams['w_d0'] + sinoparams['Delta_dw']
    w_1_roi = w_0_roi + (sinoparams['N_dw'] - 2) * sinoparams['Delta_dw']

    z_min_roi = max(w_0_roi * (-R_roi - sinoparams['u_s']) / (sinoparams['u_d0'] - sinoparams['u_s']),
                    w_0_roi * (R_roi - sinoparams['u_s']) / (sinoparams['u_d0'] - sinoparams['u_s']))

    z_max_roi = min(w_1_roi * (-R_roi - sinoparams['u_s']) / (sinoparams['u_d0'] - sinoparams['u_s']),
                    w_1_roi * (R_roi - sinoparams['u_s']) / (sinoparams['u_d0'] - sinoparams['u_s']))

    N_x_roi = 2 * math.floor(R_roi / imgparams['Delta_xy']) + 1
    N_y_roi = N_x_roi

    # imgparams['N_x'] is odd and N_x_roi is odd, imgparams['N_x'] - N_x_roi is even.
    imgparams['j_xstart_roi'] = int((imgparams['N_x'] - N_x_roi) / 2)
    imgparams['j_ystart_roi'] = imgparams['j_xstart_roi']

    imgparams['j_xstop_roi'] = int(imgparams['j_xstart_roi'] + N_x_roi - 1)
    imgparams['j_ystop_roi'] = imgparams['j_xstop_roi']

    imgparams['j_zstart_roi'] = round((z_min_roi - imgparams['z_0']) / imgparams['Delta_z'])
    imgparams['j_zstop_roi'] = imgparams['j_zstart_roi'] + round((z_max_roi - z_min_roi) / imgparams['Delta_z'])

    # Internally set by C code
    imgparams['N_x_roi'] = -1
    imgparams['N_y_roi'] = -1
    imgparams['N_z_roi'] = -1

    return imgparams

def compute_sino_params_lamino(num_views, num_det_rows, num_det_channels,
                    theta,
                    channel_offset=0.0,
                    delta_pixel_detector=1.0):
  
    # This needs to be large
    dist_factor = 500
    dist_source_detector = (dist_factor) * (delta_pixel_detector) * max(num_det_channels, num_det_rows)**2

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

    dist_dv_to_detector_center_from_source_detector_line = - channel_offset
    dist_dw_to_detector_center_from_source_detector_line = - dist_source_detector / math.tan(theta)

    # corner of detector from source-detector-line
    sinoparams[
        'v_d0'] = dist_dv_to_detector_corner_from_detector_center + dist_dv_to_detector_center_from_source_detector_line
    sinoparams[
        'w_d0'] = dist_dw_to_detector_corner_from_detector_center + dist_dw_to_detector_center_from_source_detector_line

    sinoparams['weightScaler_value'] = -1

    return sinoparams

def compute_img_params_lamino(sinoparams, theta, delta_pixel_image=None, ror_radius=None):
  
    s = delta_pixel_image
    ell = delta_pixel_image

    imgparams = dict()
    imgparams['Delta_xy'] = s
    imgparams['Delta_z'] = ell

    N_C = sinoparams['N_dv']
    T_C = sinoparams['Delta_dv']
    N_R = sinoparams['N_dw']
    T_R = sinoparams['Delta_dw']

    R = (N_R * T_R) / (2 * np.cos(theta))
    H = (N_R * T_R) / (2 * np.sin(theta))

    y_0 = sinoparams['v_d0'] + (N_C * T_C / 2)
    r_cyl_1 = (N_C * T_C / 2) - np.abs(y_0)
    r_cyl_2 = (N_R * T_R / 2)
    r = min(r_cyl_1,r_cyl_2)

    h = H - (r / np.tan(theta))

    imgparams['x_0'] = - R - s / 2
    imgparams['y_0'] = imgparams['x_0']
    imgparams['z_0'] = - H + sinoparams['w_d0'] + (N_R * T_R / 2)

    imgparams['N_x'] = int(2 * np.ceil(R / s) + 1)
    imgparams['N_y'] = imgparams['N_x']
    imgparams['N_z'] = int(2 * np.ceil(H / ell) + 1)

    imgparams['j_xstart_roi'] = int(np.ceil(R/s) - np.floor(r/s))
    imgparams['j_ystart_roi'] = imgparams['j_xstart_roi']
    imgparams['j_zstart_roi'] = int(np.ceil(H/ell) - np.floor(h/ell))

    imgparams['j_xstop_roi'] = int(np.ceil(R/s) + np.floor(r/s))
    imgparams['j_ystop_roi'] = imgparams['j_xstop_roi']
    imgparams['j_zstop_roi'] = int(np.ceil(H/ell) + np.floor(h/ell))

    imgparams['N_x_roi'] = -1
    imgparams['N_y_roi'] = -1
    imgparams['N_z_roi'] = -1

    return imgparams

def compute_img_size(num_views, num_det_rows, num_det_channels,
                     dist_source_detector,
                     magnification,
                     channel_offset=0.0, row_offset=0.0, rotation_offset=0.0,
                     delta_pixel_detector=1.0, delta_pixel_image=None, ror_radius=None,
                     geometry='cone', theta=math.pi/2):
    """Compute size of the reconstructed image given the geometric parameters.

    Args:
        num_views (int): Number of views in sinogram data
        num_det_rows (int): Number of rows in sinogram data
        num_det_channels (int): Number of channels in sinogram data

        dist_source_detector (float): Distance between the X-ray source and the detector in units of ALU
        magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).

        channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a row.
        row_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a column.
        rotation_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from source-detector line to axis of rotation in the object space.
            This is normally set to zero.
        delta_pixel_detector (float, optional): [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.
        delta_pixel_image (float, optional): [Default=None] Scalar value of image pixel spacing in :math:`ALU`.
            If None, automatically set to delta_pixel_detector/magnification
        ror_radius (float, optional): [Default=None] Scalar value of radius of reconstruction in :math:`ALU`.
            If None, automatically set.
            Pixels outside the radius ror_radius in the :math:`(x,y)` plane are disregarded in the reconstruction.

        geometry (string, optional): This can be 'cone' or 'lamino'.
        theta (float, optional): Laminographic angle, if geometry=='lamino'. For default cone beam reconstruction this is unused.

    Returns:
        Information about the image size.

        - **ROR (list)**: Region of reconstruction that specifies the size of the reconstructed image. A list of 3 integer, [num_img_slices, num_img_rows, num_img_cols]. However, the valid region of interest (ROI) is a subset of ROR.
        - **boundary_size (list)**: Number of invalid pixels on each side of a 3D image. A list of 3 integer, [img_slices_boundary_size, img_rows_boundary_size, img_cols_boundary_size]. The function `cone3D.extract_roi_from_ror` can be used to extract ROI from the full ROR.


    """
    
    # Automatically set delta_pixel_image.
    if geometry=='lamino':
        magnification = 1
    if delta_pixel_image is None:
        delta_pixel_image = delta_pixel_detector / magnification

    if geometry=='cone':
        sinoparams = compute_sino_params(dist_source_detector, magnification,
                                       num_views=num_views, num_det_rows=num_det_rows, num_det_channels=num_det_channels,
                                       channel_offset=channel_offset, row_offset=row_offset,
                                       rotation_offset=rotation_offset,
                                       delta_pixel_detector=delta_pixel_detector)

        imgparams = compute_img_params(sinoparams, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius)
    elif geometry=='lamino':
        sinoparams = compute_sino_params_lamino(num_views, num_det_rows, num_det_channels,
                                       theta,
                                       channel_offset=channel_offset,
                                       delta_pixel_detector=delta_pixel_detector)

        imgparams = compute_img_params_lamino(sinoparams, theta, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius)
    else:
        raise Exception("geometry: undefined geometry {}".format(geometry))

    # Summarize Information about the image size.
    ROR = [imgparams['N_z'], imgparams['N_x'], imgparams['N_y']]
    boundary_size = [max(imgparams['j_zstart_roi'], imgparams['N_z']-1-imgparams['j_zstop_roi']), imgparams['j_xstart_roi'], imgparams['j_ystart_roi']]

    return ROR, boundary_size


def pad_roi2ror(image, boundary_size):
    """Given a 3D ROI and the boundary size, pad the ROI with 0s to form an ROR.

    Args:
        image (ndarray): 3D numpy array with a shape of (num_img_slices, num_img_rows, num_img_cols)
        boundary_size (list): Number of invalid pixels on each side of a 3D image. A list of 3 integer, [img_slices_boundary_size, img_rows_boundary_size, img_cols_boundary_size].

    Returns:
        padded_image: 3D padded image with shape (num_img_slices+2*boundary_size[0], num_img_rows+2*boundary_size[1], num_img_cols+2*boundary_size[2]).
    """
    img_slices_boundary_size, img_rows_boundary_size, img_cols_boundary_size = boundary_size
    return np.pad(image, ((img_slices_boundary_size, img_slices_boundary_size),
                          (img_rows_boundary_size, img_rows_boundary_size),
                          (img_cols_boundary_size, img_cols_boundary_size)), mode='constant', constant_values=0)


def extract_roi_from_ror(image, boundary_size):
    """Given a 3D ROR and the boundary size, extract the ROI by removing the boundary.

    Args:
        image (ndarray): 3D numpy array with a shape of (num_img_slices, num_img_rows, num_img_cols)
        boundary_size (list): Number of invalid pixels on each side of a 3D image. A list of 3 integer, [img_slices_boundary_size, img_rows_boundary_size, img_cols_boundary_size].

    Returns:
        roi_image: 3D padded image with shape (num_img_slices-2*boundary_size[0], num_img_rows-2*boundary_size[1], num_img_cols-2*boundary_size[2]).
    """
    num_img_slices, num_img_rows, num_img_cols = image.shape
    img_slices_boundary_size, img_rows_boundary_size, img_cols_boundary_size = boundary_size

    assert num_img_slices > 2 * img_slices_boundary_size and num_img_slices > 2 * img_slices_boundary_size and num_img_slices > 2 * img_slices_boundary_size, 'The shape of the roi image should be positive.'
    return image[img_slices_boundary_size:-img_slices_boundary_size,
           img_rows_boundary_size:-img_rows_boundary_size,
           img_cols_boundary_size:-img_cols_boundary_size]


def recon(sino, angles, dist_source_detector, magnification,
          channel_offset=0.0, row_offset=0.0, rotation_offset=0.0,
          delta_pixel_detector=1.0, delta_pixel_image=None, ror_radius=None,
          geometry='cone', theta=math.pi/2,
          init_image=0.0, prox_image=None,
          sigma_y=None, snr_db=40.0, weights=None, weight_type='unweighted',
          positivity=True, p=1.2, q=2.0, T=1.0, num_neighbors=6,
          sharpness=0.0, sigma_x=None, sigma_p=None, max_iterations=100, stop_threshold=0.02,
          num_threads=None, NHICD=False, verbose=1, lib_path=__lib_path):
    """Compute 3D cone beam MBIR reconstruction
    
    Args:
        sino (ndarray): 3D sinogram array with shape (num_views, num_det_rows, num_det_channels)
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
        ror_radius (float, optional): [Default=None] Scalar value of radius of reconstruction in :math:`ALU`.
            If None, automatically set with compute_img_params.
            Pixels outside the radius ror_radius in the :math:`(x,y)` plane are disregarded in the reconstruction.
        
        geometry (string, optional): This can be 'cone' or 'lamino'.
        theta (float, optional): Laminographic angle, if geometry=='lamino'. For default cone beam reconstruction this is unused.
        
        init_image (ndarray, optional): [Default=0.0] Initial value of reconstruction image, specified by either a scalar value or a 3D numpy array with shape (num_img_slices,num_img_rows,num_img_cols)
        prox_image (ndarray, optional): [Default=None] 3D proximal map input image. 3D numpy array with shape (num_img_slices,num_img_rows,num_img_cols)
        
        sigma_y (float, optional): [Default=None] Scalar value of noise standard deviation parameter.
            If None, automatically set with auto_sigma_y.
        snr_db (float, optional): [Default=40.0] Scalar value that controls assumed signal-to-noise ratio of the data in dB.
            Ignored if sigma_y is not None.
        weights (ndarray, optional): [Default=None] 3D weights array with same shape as sino.
        weight_type (string, optional): [Default='unweighted'] Type of noise model used for data.
            If the ``weights`` array is not supplied, then the function ``cone3D.calc_weights`` is used to set weights using specified ``weight_type`` parameter.
            
                - Option "unweighted" corresponds to unweighted reconstruction;
                - Option "transmission" is the correct weighting for transmission CT with constant dosage;
                - Option "transmission_root" is commonly used with transmission CT data to improve image homogeneity;
                - Option "emission" is appropriate for emission CT data.

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
            Ignored if ``sigma_x`` is not None in qGGMRF mode, or if ``sigma_p`` is not None in proximal map mode.
        sigma_x (float, optional): [Default=None] Scalar value :math:`>0` that specifies the qGGMRF scale parameter.
            Ignored if prox_image is not None.
            If None and prox_image is also None, automatically set with auto_sigma_x. Regularization should be controled with the ``sharpness`` parameter, but ``sigma_x`` can be set directly by expert users.
        sigma_p (float, optional): [Default=None] Scalar value :math:`>0` that specifies the proximal map parameter.
            Ignored if prox_image is None.
            If None and proximal image is not None, automatically set with auto_sigma_p. Regularization should be controled with the ``sharpness`` parameter, but ``sigma_p`` can be set directly by expert users.
        max_iterations (int, optional): [Default=100] Integer valued specifying the maximum number of iterations.
        stop_threshold (float, optional): [Default=0.02] Scalar valued stopping threshold in percent.
            If stop_threshold=0.0, then run max iterations.
        num_threads (int, optional): [Default=None] Number of compute threads requested when executed.
            If None, num_threads is set to the number of cores in the system
        NHICD (bool, optional): [Default=False] If true, uses Non-homogeneous ICD updates
        verbose (int, optional): [Default=1] Possible values are {0,1,2}, where 0 is quiet, 1 prints minimal reconstruction progress information, and 2 prints the full information.
        lib_path (str, optional): [Default=~/.cache/mbircone] Path to directory containing library of forward projection matrices.
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

    if geometry=='lamino':
        magnification = 1
    if delta_pixel_image is None:
        delta_pixel_image = delta_pixel_detector / magnification

    (num_views, num_det_rows, num_det_channels) = sino.shape

    if geometry=='cone':
        sinoparams = compute_sino_params(dist_source_detector, magnification,
                                       num_views=num_views, num_det_rows=num_det_rows, num_det_channels=num_det_channels,
                                       channel_offset=channel_offset, row_offset=row_offset,
                                       rotation_offset=rotation_offset,
                                       delta_pixel_detector=delta_pixel_detector)

        imgparams = compute_img_params(sinoparams, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius)
    elif geometry=='lamino':
        sinoparams = compute_sino_params_lamino(num_views, num_det_rows, num_det_channels,
                                       theta,
                                       channel_offset=channel_offset,
                                       delta_pixel_detector=delta_pixel_detector)

        imgparams = compute_img_params_lamino(sinoparams, theta, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius)
    else:
        raise Exception("geometry: undefined geometry {}".format(geometry))

    hash_val = hash_params(angles, sinoparams, imgparams)
    sysmatrix_fname = _gen_sysmatrix_fname(lib_path=lib_path, sysmatrix_name=hash_val[:__namelen_sysmatrix])

    if os.path.exists(sysmatrix_fname):
        os.utime(sysmatrix_fname)  # update file modified time
    else:
        sysmatrix_fname_tmp = _gen_sysmatrix_fname_tmp(lib_path=lib_path, sysmatrix_name=hash_val[:__namelen_sysmatrix])
        ci.AmatrixComputeToFile_cy(angles, sinoparams, imgparams, sysmatrix_fname_tmp, verbose=verbose)
        os.rename(sysmatrix_fname_tmp, sysmatrix_fname)
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

    x = ci.recon_cy(sino, weights, init_image, prox_image,
                    sinoparams, imgparams, reconparams, sysmatrix_fname, num_threads)
    return x


def project(image, angles,
            num_det_rows, num_det_channels,
            dist_source_detector, magnification,
            channel_offset=0.0, row_offset=0.0, rotation_offset=0.0,
            delta_pixel_detector=1.0, delta_pixel_image=None, ror_radius=None,
            geometry='cone', theta=math.pi/2,
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
        
        channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a row.
        row_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a column.
        rotation_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from source-detector line to axis of rotation in the object space.
            This is normally set to zero.
        
        delta_pixel_detector (float, optional): [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.
        delta_pixel_image (float, optional): [Default=None] Scalar value of image pixel spacing in :math:`ALU`.
            If None, automatically set to delta_pixel_detector/magnification
        ror_radius (float, optional): [Default=None] Scalar value of radius of reconstruction in :math:`ALU`.
            If None, automatically set with compute_img_params.
            Pixels outside the radius ror_radius in the :math:`(x,y)` plane are disregarded.
        
        geometry (string, optional): This can be 'cone' or 'lamino'.
        theta (float, optional): Laminographic angle, if geometry=='lamino'. For default cone beam reconstruction this is unused.
        
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

    if geometry=='lamino':
        magnification = 1
    if delta_pixel_image is None:
        delta_pixel_image = delta_pixel_detector / magnification

    num_views = len(angles)

    if geometry=='cone':
        sinoparams = compute_sino_params(dist_source_detector, magnification,
                                       num_views=num_views, num_det_rows=num_det_rows, num_det_channels=num_det_channels,
                                       channel_offset=channel_offset, row_offset=row_offset,
                                       rotation_offset=rotation_offset,
                                       delta_pixel_detector=delta_pixel_detector)

        imgparams = compute_img_params(sinoparams, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius)
    elif geometry=='lamino':
        sinoparams = compute_sino_params_lamino(num_views, num_det_rows, num_det_channels,
                                       theta,
                                       channel_offset=channel_offset,
                                       delta_pixel_detector=delta_pixel_detector)

        imgparams = compute_img_params_lamino(sinoparams, theta, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius)
    else:
        raise Exception("geometry: undefined geometry {}".format(geometry))

    (num_img_slices, num_img_rows, num_img_cols) = image.shape

    assert (num_img_slices, num_img_rows, num_img_cols) == (imgparams['N_z'], imgparams['N_x'], imgparams['N_y']), \
        'Image size of %s is incorrect! With the specified geometric parameters, expected image should have shape %s, use function `cone3D.compute_img_size` to compute the correct image size.' \
        %  ((num_img_slices, num_img_rows, num_img_cols), (imgparams['N_z'], imgparams['N_x'], imgparams['N_y']))

    hash_val = hash_params(angles, sinoparams, imgparams)
    sysmatrix_fname = _gen_sysmatrix_fname(lib_path=lib_path, sysmatrix_name=hash_val[:__namelen_sysmatrix])

    if os.path.exists(sysmatrix_fname):
        os.utime(sysmatrix_fname)  # update file modified time
    else:
        sysmatrix_fname_tmp = _gen_sysmatrix_fname_tmp(lib_path=lib_path, sysmatrix_name=hash_val[:__namelen_sysmatrix])
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

