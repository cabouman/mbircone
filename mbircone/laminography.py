import numpy as np
import os
import mbircone.cone3D as cone3D

__lib_path = os.path.join(os.path.expanduser('~'), '.cache', 'mbircone')


def recon_lamino(sino, angles, theta,
                 weights=None, weight_type='unweighted', init_image=0.0, prox_image=None,
                 num_image_rows=None, num_image_cols=None, num_image_slices=None,
                 delta_det_channel=None, delta_det_row=None, delta_pixel_image=None,
                 det_channel_offset=0.0, image_slice_offset=0.0,
                 sigma_y=None, snr_db=40.0, sigma_x=None, sigma_p=None, p=1.2, q=2.0, T=1.0, num_neighbors=6,
                 sharpness=0.0, positivity=True, max_resolutions=None, stop_threshold=0.02, max_iterations=100,
                 NHICD=False, num_threads=None, verbose=1, lib_path=__lib_path):
    """ Compute 3D cone beam MBIR reconstruction for the laminography case.

    Args:
        sino (float, ndarray): 3D laminography sinogram data with shape (num_views, num_det_rows, num_det_channels).
        angles (float, ndarray): 1D array of view angles in radians.
        theta (float): Angle that source-detector line makes with the object vertical axis.
            Equal to pi/2 - grazing angle. When theta=pi/2, this is a parallel beam reconstruction.

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

        delta_det_channel (float, optional): [Default=1.0] Detector channel spacing in :math:`ALU`. If None, set to
            value of delta_det_row. If delta_det_row is None, set both to 1.0.
        delta_det_row (float, optional): [Default=1.0] Detector row spacing in :math:`ALU`. If None, set to
            value of delta_det_channel. If delta_det_channel is None, set both to 1.0.
        delta_pixel_image (float, optional): [Default=None] Image pixel spacing in :math:`ALU`.
            If None, automatically set to ``delta_det_channel/magnification``.

        det_channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector
            to the source-detector line along a row. (Note: There is no ``det_row_offset`` parameter; due to the
            parallel beam geometry such a parameter would not have an effect.)
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
        (float, ndarray): 3D laminography reconstruction image with shape (num_img_slices, num_img_rows, num_img_cols)
        in units of :math:`ALU^{-1}`.
    """

    (_, num_det_rows, num_det_channels) = sino.shape

    # If one of them is None, set it to equal the other.
    if delta_det_channel is None:
        delta_det_channel = delta_det_row
    if delta_det_row is None:
        delta_det_row = delta_det_channel

    # If both of them were None, set both to 1.0
    if delta_det_channel is None:
        delta_det_channel = 1.0
        delta_det_row = delta_det_channel

    # Copy parameters that are the same in both laminography and cone-beam coordinates

    _sino = np.copy(sino)
    _angles = np.copy(angles)

    if weights is None:
        _weights = None
    else:
        _weights = np.copy(weights)

    if isinstance(init_image, float):
        _init_image = init_image
    else:
        _init_image = np.copy(init_image)

    if prox_image is None:
        _prox_image = None
    else:
        _prox_image = np.copy(prox_image)

    _weight_type = weight_type
    _num_image_rows = num_image_rows
    _num_image_cols = num_image_cols
    _num_image_slices = num_image_slices
    _delta_det_channel = delta_det_channel
    _delta_pixel_image = delta_pixel_image
    _det_channel_offset = det_channel_offset

    _sigma_y = sigma_y
    _snr_db = snr_db
    _sigma_x = sigma_x
    _sigma_p = sigma_p
    _p = p
    _q = q
    _T = T

    _num_neighbors = num_neighbors
    _sharpness = sharpness
    _positivity = positivity
    _max_resolutions = max_resolutions
    _stop_threshold = stop_threshold
    _max_iterations = max_iterations
    _NHICD = NHICD
    _num_threads = num_threads
    _verbose = verbose
    _lib_path = lib_path

    # Calculate parameters that differ when we switch to synthetic cone-beam coordinates

    # Determine synthetic source-detector distance to approximate parallel beams
    # Beams should spread by less than epsilon * max(delta_det_channel,delta_det_row) as they pass through the phantom
    epsilon = 0.0015
    _dist_source_detector = (1/epsilon)*max(delta_det_channel, delta_det_row)*(max(num_det_rows, num_det_channels)**2)

    # Setting _magnification=1.0 with a large _dist_source_detector approximates parallel beams
    _magnification = 1.0

    # Compute synthetic detector pixel size corresponding to affine projection of
    # real parallel-beam laminography detector onto synthetic cone-beam detector
    _delta_det_row = delta_det_row / np.sin(theta)

    # Move synthetic cone-beam detector downward so that the incident cone-beam angle corresponds
    # to the real laminographic angle.
    _det_row_offset = _dist_source_detector / np.tan(theta)

    # Since there is no source-detector line in parallel-beam projection, define the
    # synthetic source-detector line to intersect the axis of rotation; so _rotation_offset=0.0
    _rotation_offset = 0.0

    # Move the image region to correspond with the movement of the synthetic cone-beam detector
    _image_slice_offset = image_slice_offset + _det_row_offset

    # Translate laminography geometry to cone-beam approximation of laminography
    return cone3D.recon(_sino, _angles, dist_source_detector=_dist_source_detector, magnification=_magnification,
                        weights=_weights, weight_type=_weight_type, init_image=_init_image, prox_image=_prox_image,
                        num_image_rows=_num_image_rows, num_image_cols=_num_image_cols,
                        num_image_slices=_num_image_slices,
                        delta_det_channel=_delta_det_channel, delta_det_row=_delta_det_row,
                        delta_pixel_image=_delta_pixel_image,
                        det_channel_offset=_det_channel_offset, det_row_offset=_det_row_offset,
                        rotation_offset=_rotation_offset, image_slice_offset=_image_slice_offset,
                        sigma_y=_sigma_y, snr_db=_snr_db, sigma_x=_sigma_x, sigma_p=_sigma_p, p=_p, q=_q, T=_T,
                        num_neighbors=_num_neighbors,
                        sharpness=_sharpness, positivity=_positivity, max_resolutions=_max_resolutions,
                        stop_threshold=_stop_threshold, max_iterations=_max_iterations,
                        NHICD=_NHICD, num_threads=_num_threads, verbose=_verbose, lib_path=_lib_path)


def project_lamino(image, angles, theta,
                   num_det_rows, num_det_channels,
                   delta_det_channel=None, delta_det_row=None, delta_pixel_image=None,
                   det_channel_offset=0.0, image_slice_offset=0.0,
                   num_threads=None, verbose=1, lib_path=__lib_path):
    """ Compute 3D cone beam forward projection for the laminography case.

    Args:
        image (float, ndarray): 3D image to be projected, with shape (num_img_slices, num_img_rows, num_img_cols).
        angles (float, ndarray): 1D array of view angles in radians.
        theta (float): Angle that source-detector line makes with the object vertical axis.
            Equal to pi/2 - grazing angle. When theta=pi/2, this is a parallel beam projection.

        num_det_rows (int): Number of rows in laminography sinogram data.
        num_det_channels (int): Number of channels in laminography sinogram data.

        delta_det_channel (float, optional): [Default=1.0] Detector channel spacing in :math:`ALU`. If None, set to
            value of delta_det_row. If delta_det_row is None, set both to 1.0.
        delta_det_row (float, optional): [Default=1.0] Detector row spacing in :math:`ALU`. If None, set to
            value of delta_det_channel. If delta_det_channel is None, set both to 1.0.
        delta_pixel_image (float, optional): [Default=None] Image pixel spacing in :math:`ALU`.
            If None, automatically set to ``delta_det_channel/magnification``.

        det_channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector
            to the source-detector line along a row. (Note: There is no ``det_row_offset`` parameter; due to the
            parallel beam geometry such a parameter would not have an effect.)
        image_slice_offset (float, optional): [Default=0.0] Vertical offset of the image in units of :math:`ALU`.

        num_threads (int, optional): [Default=None] Number of compute threads requested when executed.
            If None, ``num_threads`` is set to the number of cores in the system.
        verbose (int, optional): [Default=1] Possible values are {0,1,2}, where 0 is quiet, 1 prints minimal
            reconstruction progress information, and 2 prints the full information.
        lib_path (str, optional): [Default=~/.cache/mbircone] Path to directory containing library of
            forward projection matrices.

    Returns:
        (float, ndarray): 3D laminography sinogram with shape (num_views, num_det_rows, num_det_channels).
    """

    # If one of them is None, set it to equal the other.
    if delta_det_channel is None:
        delta_det_channel = delta_det_row
    if delta_det_row is None:
        delta_det_row = delta_det_channel

    # If both of them were None, set both to 1.0
    if delta_det_channel is None:
        delta_det_channel = 1.0
        delta_det_row = delta_det_channel

    # Copy parameters that are the same in both laminography and cone-beam coordinates
    _image = np.copy(image)
    _angles = np.copy(angles)

    _num_det_rows = num_det_rows
    _num_det_channels = num_det_channels
    _delta_det_channel = delta_det_channel
    _delta_pixel_image = delta_pixel_image
    _det_channel_offset = det_channel_offset

    _num_threads = num_threads
    _verbose = verbose
    _lib_path = lib_path

    # Determine artificial source-detector distance to approximate parallel beams
    # Beams should spread by less than epsilon * max(delta_det_channel,delta_det_row) as they pass through the phantom
    epsilon = 0.0015
    _dist_source_detector = (1/epsilon)*max(delta_det_channel,delta_det_row)*(max(num_det_rows, num_det_channels)**2)

    # Setting _magnification=1.0 with a large _dist_source_detector approximates parallel beams
    _magnification = 1.0

    # Compute synthetic detector pixel size corresponding to affine projection of
    # real parallel-beam laminography detector onto synthetic cone-beam detector
    _delta_det_row = delta_det_row/np.sin(theta)

    # Move synthetic cone-beam detector downward so that the incident cone-beam angle corresponds
    # to the real laminographic angle.
    _det_row_offset = _dist_source_detector/np.tan(theta)

    # Since there is no source-detector line in parallel-beam projection, define the
    # synthetic source-detector line to intersect the axis of rotation; so _rotation_offset=0.0
    _rotation_offset = 0.0

    # Move the image region to correspond with the movement of the synthetic cone-beam detector
    _image_slice_offset = image_slice_offset + _det_row_offset

    return cone3D.project(image=_image, angles=_angles,
                          num_det_rows=_num_det_rows, num_det_channels=_num_det_channels,
                          dist_source_detector=_dist_source_detector, magnification=_magnification,
                          delta_det_channel=_delta_det_channel, delta_det_row=_delta_det_row,
                          delta_pixel_image=_delta_pixel_image,
                          det_channel_offset=_det_channel_offset, det_row_offset=_det_row_offset,
                          rotation_offset=_rotation_offset,
                          image_slice_offset=_image_slice_offset,
                          num_threads=_num_threads, verbose=_verbose, lib_path=_lib_path)
