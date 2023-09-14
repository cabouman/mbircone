import numpy as np
import os
import warnings
import mbircone.cone3D as cone3D
from docstring_inheritance import inherit_google_docstring

__lib_path = os.path.join(os.path.expanduser('~'), '.cache', 'mbircone')


def auto_lamino_params(theta, num_det_rows, num_det_channels, delta_det_row, delta_det_channel, image_slice_offset):
    """ Compute values for parameters used internally for a synthetic cone beam approximation of laminography.

    Args:
        theta (float): Angle in radians that source-detector line makes with the object vertical axis.
        num_det_rows (int): Number of rows in laminography sinogram data.
        num_det_channels (int): Number of channels in laminography sinogram data.

        delta_det_row (float): Detector row spacing in :math:`ALU`.
        delta_det_channel (float): Detector channel spacing in :math:`ALU`.

        image_slice_offset (float): Vertical offset of the image in units of :math:`ALU`.

    Returns:
        (float, tuple): Values for ``lamino_dist_source_detector``, ``lamino_magnification``,
        ``lamino_delta_det_row`` ``lamino_det_row_offset``, ``lamino_rotation_offset``,
        ``lamino_image_slice_offset`` for the inputted image measurements.
    """

    # Determine artificial source-detector distance to approximate parallel beams
    # Beams spread by less than epsilon * max(delta_det_channel,delta_det_row) as they pass through the phantom
    epsilon = 0.0015
    lamino_dist_source_detector = (1 / epsilon) * max(delta_det_channel, delta_det_row) * \
                                  (max(num_det_rows, num_det_channels) ** 2)

    # Setting lamino_magnification=1.0 with a large lamino_dist_source_detector approximates parallel beams
    lamino_magnification = 1.0

    # Compute synthetic detector pixel size corresponding to affine projection of
    # real parallel-beam laminography detector onto synthetic cone-beam detector
    lamino_delta_det_row = delta_det_row / np.sin(theta)

    # Move synthetic cone-beam detector downward so that the incident cone-beam angle corresponds
    # to the real laminographic angle.
    lamino_det_row_offset = lamino_dist_source_detector / np.tan(theta)

    # Since there is no source-detector line in parallel-beam projection, define the
    # synthetic source-detector line to intersect the axis of rotation; so lamino_rotation_offset=0.0
    lamino_rotation_offset = 0.0

    # Move the image region to correspond with the movement of the synthetic cone-beam detector
    lamino_image_slice_offset = image_slice_offset + lamino_det_row_offset

    return lamino_dist_source_detector, lamino_magnification, lamino_delta_det_row, lamino_det_row_offset, \
        lamino_rotation_offset, lamino_image_slice_offset


def auto_image_slices(theta, num_det_rows, num_det_channels, delta_det_row, delta_det_channel, delta_pixel_image):
    """ Compute the automatic image array slice dimension for use in qGGMRF reconstruction.

    Args:
        theta (float): Angle that source-detector line makes with the object vertical axis, in radians.

        num_det_rows (int): Number of rows in laminography sinogram data.
        num_det_channels (int): Number of channels in laminography sinogram data.

        delta_det_row (float): Detector row spacing in :math:`ALU`.
        delta_det_channel (float): Detector channel spacing in :math:`ALU`.

        delta_pixel_image (float): Image pixel spacing in :math:`ALU`.

    Returns:
        (int): Default value for ``num_image_slices`` for the inputted detector measurements.
    """

    if (num_det_rows * delta_det_row) > (num_det_channels * delta_det_channel) * np.cos(theta):
        # Set region of reconstruction to cylinder inside double-cone
        image_thickness = ((num_det_rows * delta_det_row) / np.sin(theta)) - \
                          ((num_det_channels * delta_det_channel) / np.tan(theta))
    else:
        # Set region of reconstruction to double-cone
        image_thickness = (num_det_rows * delta_det_row) / np.sin(theta)

    # Convert absolute measurements to pixel measurements
    num_image_slices = int(np.ceil(image_thickness / delta_pixel_image))

    return num_image_slices


def auto_image_rows_cols(theta, num_det_rows, num_det_channels, delta_det_row, delta_det_channel, num_image_slices,
                         delta_pixel_image):
    """ Compute the automatic image array row and col dimensions for use in qGGMRF reconstruction.

    Args:
        theta (float): Angle that source-detector line makes with the object vertical axis, in radians.

        num_det_rows (int): Number of rows in laminography sinogram data.
        num_det_channels (int): Number of channels in laminography sinogram data.

        delta_det_row (float): Detector row spacing in :math:`ALU`.
        delta_det_channel (float): Detector channel spacing in :math:`ALU`.

        num_image_slices (int): Number of slices in reconstructed image.

        delta_pixel_image (float): Image pixel spacing in :math:`ALU`.

    Returns:
        (int): (int, 2-tuple): Default values for ``num_image_rows``, ``num_image_cols`` for the
        inputted detector measurements.
    """

    detector_height = num_det_rows * delta_det_row
    detector_width = num_det_channels * delta_det_channel

    # Compute diagonal of image that detector makes on a given horizontal slice for fixed z
    detector_image_diagonal = np.sqrt((detector_width) ** 2 + (detector_height / np.cos(theta)) ** 2)

    # Compute image rows and columns
    num_image_rows = int(np.ceil( (detector_image_diagonal/delta_pixel_image) + (num_image_slices * np.tan(theta)) ))
    num_image_cols = int(np.ceil( (detector_image_diagonal/delta_pixel_image) + (num_image_slices * np.tan(theta)) ))

    return num_image_rows, num_image_cols


def recon_lamino(sino, angles, theta,
                 weights=None, weight_type='unweighted', init_image=0.0, prox_image=None,
                 num_image_rows=None, num_image_cols=None, num_image_slices=None,
                 delta_det_channel=1.0, delta_det_row=1.0, delta_pixel_image=None,
                 det_channel_offset=0.0, image_slice_offset=0.0,
                 sigma_y=None, snr_db=40.0, sigma_x=None, sigma_p=None, p=1.2, q=2.0, T=1.0, num_neighbors=6,
                 sharpness=0.0, positivity=True, max_resolutions=None, stop_threshold=0.20, max_iterations=100,
                 NHICD=False, num_threads=None, verbose=1, lib_path=__lib_path):
    """Compute MBIR reconstruction for parallel-beam laminography geometry.

    Args:
        sino (float, ndarray): 3D laminography sinogram data with shape (num_views, num_det_rows, num_det_channels).
        theta (float): Angle in radians that source-detector line makes with the object vertical axis.

        num_image_rows (int, optional): [Default=None] Number of rows in reconstructed image.
            If None, automatically set by ``laminography.auto_image_rows_cols``.
        num_image_cols (int, optional): [Default=None] Number of columns in reconstructed image.
            If None, automatically set by ``laminography.auto_image_rows_cols``.
        num_image_slices (int, optional): [Default=None] Number of slices in reconstructed image.
            If None, automatically set by ``laminography.auto_image_slices``.

        delta_pixel_image (float, optional): [Default=None] Image pixel spacing in :math:`ALU`.
            If None, automatically set to ``delta_det_channel``.

        det_channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector
            to the detector z-axis along a row. (Note: There is no ``det_row_offset`` parameter; due to the
            parallel beam geometry such a parameter would be redundant with image_slice_offset.)

    Returns:
        (float, ndarray): 3D laminography reconstruction image with shape (num_img_slices, num_img_rows, num_img_cols)
        in units of :math:`ALU^{-1}`.

    """

    (_, num_det_rows, num_det_channels) = sino.shape

    if delta_pixel_image is None:
        delta_pixel_image = delta_det_channel

    # Set image parameters automatically
    if num_image_slices is None:
        num_image_slices = auto_image_slices(theta, num_det_rows, num_det_channels, delta_det_row,
                                             delta_det_channel, delta_pixel_image)
        warnings.warn(f'\n*** Parameter num_image_slices was not given. Setting to {num_image_slices}. '
                      'A smaller value may speed up qGGMRF reconstruction. ***\n')
    if num_image_rows is None:
        num_image_rows, _ = auto_image_rows_cols(theta, num_det_rows, num_det_channels, delta_det_row,
                                                 delta_det_channel, num_image_slices, delta_pixel_image)
        warnings.warn(f'\n*** Parameter num_image_rows was not given. Setting to {num_image_rows}. '
                      'A smaller value may speed up qGGMRF reconstruction. ***\n')
    if num_image_cols is None:
        _, num_image_cols = auto_image_rows_cols(theta, num_det_rows, num_det_channels, delta_det_row,
                                                 delta_det_channel, num_image_slices, delta_pixel_image)
        warnings.warn(f'\n*** Parameter num_image_cols was not given. Setting to {num_image_cols}. '
                      'A smaller value may speed up qGGMRF reconstruction. ***\n')

    lamino_dist_source_detector, lamino_magnification, lamino_delta_det_row, lamino_det_row_offset, \
        lamino_rotation_offset, lamino_image_slice_offset = auto_lamino_params(theta, num_det_rows, num_det_channels,
                                                                               delta_det_channel, delta_det_row,
                                                                               image_slice_offset)

    # Translate laminography geometry to cone-beam approximation of laminography
    return cone3D.recon(sino, angles, dist_source_detector=lamino_dist_source_detector,
                        magnification=lamino_magnification,
                        weights=weights, weight_type=weight_type, init_image=init_image, prox_image=prox_image,
                        num_image_rows=num_image_rows, num_image_cols=num_image_cols,
                        num_image_slices=num_image_slices,
                        delta_det_channel=delta_det_channel, delta_det_row=lamino_delta_det_row,
                        delta_pixel_image=delta_pixel_image,
                        det_channel_offset=det_channel_offset, det_row_offset=lamino_det_row_offset,
                        rotation_offset=lamino_rotation_offset, image_slice_offset=lamino_image_slice_offset,
                        sigma_y=sigma_y, snr_db=snr_db, sigma_x=sigma_x, sigma_p=sigma_p, p=p, q=q, T=T,
                        num_neighbors=num_neighbors,
                        sharpness=sharpness, positivity=positivity, max_resolutions=max_resolutions,
                        stop_threshold=stop_threshold, max_iterations=max_iterations,
                        NHICD=NHICD, num_threads=num_threads, verbose=verbose, lib_path=lib_path,
                        lamino_mode=True)


def project_lamino(image, angles, theta,
                   num_det_rows, num_det_channels,
                   delta_det_channel=1.0, delta_det_row=1.0, delta_pixel_image=None,
                   det_channel_offset=0.0, image_slice_offset=0.0,
                   num_threads=None, verbose=1, lib_path=__lib_path):
    """ Compute forward projection for parallel-beam laminography geometry.

    Args:
        theta (float): Angle in radians that source-detector line makes with the object vertical axis.

        num_det_rows (int): Number of rows in laminography sinogram data.
        num_det_channels (int): Number of channels in laminography sinogram data.

        delta_pixel_image (float, optional): [Default=None] Image pixel spacing in :math:`ALU`.
            If None, automatically set to ``delta_det_channel``.

        det_channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector
            to the detector z-axis along a row. (Note: There is no ``det_row_offset`` parameter; due to the
            parallel beam geometry such a parameter would be redundant with image_slice_offset.)

    Returns:
        (float, ndarray): 3D laminography sinogram with shape (num_views, num_det_rows, num_det_channels).

    """

    lamino_dist_source_detector, lamino_magnification, lamino_delta_det_row, lamino_det_row_offset, \
        lamino_rotation_offset, lamino_image_slice_offset = auto_lamino_params(theta, num_det_rows, num_det_channels,
                                                                               delta_det_channel, delta_det_row,
                                                                               image_slice_offset)

    return cone3D.project(image=image, angles=angles,
                          num_det_rows=num_det_rows, num_det_channels=num_det_channels,
                          dist_source_detector=lamino_dist_source_detector, magnification=lamino_magnification,
                          delta_det_channel=delta_det_channel, delta_det_row=lamino_delta_det_row,
                          delta_pixel_image=delta_pixel_image,
                          det_channel_offset=det_channel_offset, det_row_offset=lamino_det_row_offset,
                          rotation_offset=lamino_rotation_offset,
                          image_slice_offset=lamino_image_slice_offset,
                          num_threads=num_threads, verbose=verbose, lib_path=lib_path)


# Inherit Google docstrings from parent functions using docstring-inheritance package.
# Automatically compiled with Sphinx
# Documentation at https://pypi.org/project/docstring-inheritance/
inherit_google_docstring(cone3D.recon.__doc__, recon_lamino)
inherit_google_docstring(cone3D.project.__doc__, project_lamino)
