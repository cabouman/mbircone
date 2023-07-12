import os
from glob import glob
import numpy as np
from PIL import Image
import warnings
import math
from mbircone import cone3D

__lib_path = os.path.join(os.path.expanduser('~'), '.cache', 'mbircone')


def _read_scan_img(img_path):
    """Reads a single scan image from an image path.

    Args:
        img_path (string): Path to a ConeBeam scan image.
    Returns:
        ndarray (float): 2D numpy array. A single scan image.
    """

    img = np.asarray(Image.open(img_path))

    if np.issubdtype(img.dtype, np.integer):
        # make float and normalize integer types
        maxval = np.iinfo(img.dtype).max
        img = img.astype(np.float32) / maxval

    return img.astype(np.float32)


def _read_scan_dir(scan_dir, view_ids=[]):
    """Reads a stack of scan images from a directory.

    Args:
        scan_dir (string): Path to a ConeBeam Scan directory.
        view_ids (list[int]): List of view indices to specify which scans to read.
    Returns:
        ndarray (float): 3D numpy array, (num_views, num_det_rows, num_det_channels). A stack of scan images.
    """

    if view_ids == []:
        warnings.warn("view_ids should not be empty.")

    img_path_list = sorted(glob(os.path.join(scan_dir, '*')))
    img_path_list = [img_path_list[i] for i in view_ids]
    img_list = [_read_scan_img(img_path) for img_path in img_path_list]

    # return shape = num_views x num_det_rows x num_det_channels
    return np.stack(img_list, axis=0)


def _downsample_scans(obj_scan, blank_scan, dark_scan,
                      downsample_factor,
                      defective_pixel_list=None):
    """Performs Down-sampling to the scan images in the detector plane.

    Args:
        obj_scan (float): A stack of sinograms. 3D numpy array, (num_views, num_det_rows, num_det_channels).
        blank_scan (float): A blank scan. 2D numpy array, (num_det_rows, num_det_channels).
        dark_scan (float): A dark scan. 3D numpy array, (num_det_rows, num_det_channels).
        downsample_factor ([int, int]): Default=[1,1]] Two numbers to define down-sample factor.
    Returns:
        Downsampled scans
        - **obj_scan** (*ndarray, float*): A stack of sinograms. 3D numpy array, (num_views, num_det_rows, num_det_channels).
        - **blank_scan** (*ndarray, float*): A blank scan. 3D numpy array, (num_det_rows, num_det_channels).
        - **dark_scan** (*ndarray, float*): A dark scan. 3D numpy array, (num_det_rows, num_det_channels).
    """

    assert len(downsample_factor) == 2, 'factor({}) needs to be of len 2'.format(downsample_factor)
    assert (downsample_factor[0]>=1 and downsample_factor[1]>=1), 'factor({}) along each dimension should be greater or equal to 1'.format(downsample_factor)

    good_pixel_mask = np.ones((blank_scan.shape[1], blank_scan.shape[2]), dtype=int)
    if defective_pixel_list is not None:
        for (r,c) in defective_pixel_list:
            good_pixel_mask[r,c] = 0

    # crop the scan if the size is not divisible by downsample_factor.
    new_size1 = downsample_factor[0] * (obj_scan.shape[1] // downsample_factor[0])
    new_size2 = downsample_factor[1] * (obj_scan.shape[2] // downsample_factor[1])

    obj_scan = obj_scan[:, 0:new_size1, 0:new_size2]
    blank_scan = blank_scan[:, 0:new_size1, 0:new_size2]
    dark_scan = dark_scan[:, 0:new_size1, 0:new_size2]
    good_pixel_mask = good_pixel_mask[0:new_size1, 0:new_size2]

    ###### Compute block sum of the high res scan images. Defective pixels are excluded.
    # filter out defective pixels
    good_pixel_mask = good_pixel_mask.reshape(good_pixel_mask.shape[0] // downsample_factor[0], downsample_factor[0],
                                              good_pixel_mask.shape[1] // downsample_factor[1], downsample_factor[1])
    obj_scan = obj_scan.reshape(obj_scan.shape[0],
                                obj_scan.shape[1] // downsample_factor[0], downsample_factor[0],
                                obj_scan.shape[2] // downsample_factor[1], downsample_factor[1]) * good_pixel_mask

    blank_scan = blank_scan.reshape(blank_scan.shape[0],
                                    blank_scan.shape[1] // downsample_factor[0], downsample_factor[0],
                                    blank_scan.shape[2] // downsample_factor[1], downsample_factor[1]) * good_pixel_mask
    dark_scan = dark_scan.reshape(dark_scan.shape[0],
                                  dark_scan.shape[1] // downsample_factor[0], downsample_factor[0],
                                  dark_scan.shape[2] // downsample_factor[1], downsample_factor[1]) * good_pixel_mask

    # compute block sum
    obj_scan = obj_scan.sum((2,4))
    blank_scan = blank_scan.sum((2, 4))
    dark_scan = dark_scan.sum((2, 4))
    # number of good pixels in each down-sampling block
    good_pixel_count = good_pixel_mask.sum((1,3))

    # new defective pixel list = {indices of pixels where the downsampling block contains all bad pixels}
    defective_pixel_list = np.argwhere(good_pixel_count < 1)

    # compute block averaging by dividing block sum with number of good pixels in the block
    obj_scan = obj_scan / good_pixel_count
    blank_scan = blank_scan / good_pixel_count
    dark_scan = dark_scan / good_pixel_count

    return obj_scan, blank_scan, dark_scan, defective_pixel_list


def _crop_scans(obj_scan, blank_scan, dark_scan,
                crop_factor=[(0, 0), (1, 1)],
                defective_pixel_list=None):
    """Crops given scans with given factor.

    Args:
        obj_scan (float): A stack of sinograms. 3D numpy array, (num_views, num_det_rows, num_det_channels).
        blank_scan (float) : A blank scan. 3D numpy array, (1, num_det_rows, num_det_channels).
        dark_scan (float): A dark scan. 3D numpy array, (1, num_det_rows, num_det_channels).
        crop_factor ([(int, int),(int, int)] or [int, int, int, int]):
            [Default=[(0, 0), (1, 1)]] Two points to define the bounding box. Sequence of [(r0, c0), (r1, c1)] or
            [r0, c0, r1, c1], where 0<=r0 <= r1<=1 and 0<=c0 <= c1<=1.

    Returns:
        Cropped scans
        - **obj_scan** (*ndarray, float*): A stack of sinograms. 3D numpy array, (num_views, num_det_rows, num_det_channels).
        - **blank_scan** (*ndarray, float*): A blank scan. 3D numpy array, (1, num_det_rows, num_det_channels).
        - **dark_scan** (*ndarray, float*): A dark scan. 3D numpy array, (1, num_det_rows, num_det_channels).
    """
    if isinstance(crop_factor[0], (list, tuple)):
        (r0, c0), (r1, c1) = crop_factor
    else:
        r0, c0, r1, c1 = crop_factor

    assert 0 <= r0 <= r1 <= 1 and 0 <= c0 <= c1 <= 1, 'crop_factor should be sequence of [(r0, c0), (r1, c1)] ' \
                                                      'or [r0, c0, r1, c1], where 1>=r1 >= r0>=0 and 1>=c1 >= c0>=0.'
    assert math.isclose(c0, 1 - c1), 'horizontal crop limits must be symmetric'

    N1_lo = round(r0 * obj_scan.shape[1])
    N2_lo = round(c0 * obj_scan.shape[2])

    N1_hi = round(r1 * obj_scan.shape[1])
    N2_hi = round(c1 * obj_scan.shape[2])

    obj_scan = obj_scan[:, N1_lo:N1_hi, N2_lo:N2_hi]
    blank_scan = blank_scan[:, N1_lo:N1_hi, N2_lo:N2_hi]
    dark_scan = dark_scan[:, N1_lo:N1_hi, N2_lo:N2_hi]

    # adjust the defective pixel information: any down-sampling block containing a defective pixel is also defective
    if defective_pixel_list is not None:
        i = 0
        while i < len(defective_pixel_list):
            (r,c) = defective_pixel_list[i]
            (r_new, c_new) = (r-N1_lo, c-N2_lo)
            # delete the index tuple if it falls outside the cropped region
            if (r_new<0 or r_new>=obj_scan.shape[1] or c_new<0 or c_new>=obj_scan.shape[2]):
                del defective_pixel_list[i]
            else:
                i+=1
    return obj_scan, blank_scan, dark_scan, defective_pixel_list



def _NSI_read_str_from_config(filepath, tags_sections):
    """Returns strings about dataset information read from NSI configuration file.

    Args:
        filepath (string): Path to NSI configuration file. The filename extension is '.nsipro'.
        tags_sections (list[string,string]): Given tags and sections to locate the information we want to read.
    Returns:
        list[string], a list of strings have needed dataset information for reconstruction.

    """
    tag_strs = ['<' + tag + '>' for tag, section in tags_sections]
    section_starts = ['<' + section + '>' for tag, section in tags_sections]
    section_ends = ['</' + section + '>' for tag, section in tags_sections]
    NSI_params = []

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except IOError:
        print("Could not read file:", filepath)

    for tag_str, section_start, section_end in zip(tag_strs, section_starts, section_ends):
        section_start_inds = [ind for ind, match in enumerate(lines) if section_start in match]
        section_end_inds = [ind for ind, match in enumerate(lines) if section_end in match]
        section_start_ind = section_start_inds[0]
        section_end_ind = section_end_inds[0]

        for line_ind in range(section_start_ind + 1, section_end_ind):
            line = lines[line_ind]
            if tag_str in line:
                tag_ind = line.find(tag_str, 1) + len(tag_str)
                if tag_ind == -1:
                    NSI_params.append("")
                else:
                    NSI_params.append(line[tag_ind:].strip('\n'))

    return NSI_params


def NSI_load_scans_and_params(config_file_path, obj_scan_path, blank_scan_path, dark_scan_path=None,
                              defective_pixel_path=None,
                              geo_params_only=False,
                              downsample_factor=[1, 1], crop_factor=[(0, 0), (1, 1)],
                              view_id_start=0, view_angle_start=0.,
                              view_id_end=None, subsample_view_factor=1):
    """ Load the object scan, blank scan, dark scan, view angles, defective pixel information, and geometry parameters from an NSI dataset directory.

    The scan images will be (optionally) cropped and downsampled.

    A subset of the views may be selected based on user input. In that case, the object scan images and view angles corresponding to the subset of the views will be returned.

    This function is specific to NSI datasets.

    **Arguments specific to file paths**:

        - config_file_path (string): Path to NSI configuration file. The filename extension is '.nsipro'.
        - obj_scan_path (string): Path to an NSI radiograph directory.
        - blank_scan_path (string): [Default=None] Path to a blank scan image, e.g. 'dataset_path/Corrections/gain0.tif'
        - dark_scan_path (string): [Default=None] Path to a dark scan image, e.g. 'dataset_path/Corrections/offset.tif'
        - defective_pixel_path (string): [Default=None] Path to the file containing defective pixel information, e.g. 'dataset_path/Corrections/defective_pixels.defect'

        - geo_params_only (bool): [Default=False] If True, only the geo params dictionary will be returned. 
    **Arguments specific to radiograph downsampling and cropping**:

        - downsample_factor ([int, int]): [Default=[1,1]] Down-sample factors along the detector rows and channels respectively. By default no downsampling will be performed.

            In case where the scan size is not divisible by `downsample_factor`, the scans will be first truncated to a size that is divisible by `downsample_factor`, and then downsampled.

        - crop_factor ([(float, float),(float, float)] or [float, float, float, float]): [Default=[(0., 0.), (1., 1.)]]. Two fractional points [(r0, c0), (r1, c1)] defining the bounding box that crops the scans, where 0<=r0<=r1<=1 and 0<=c0<=c1<=1. By default no cropping will be performed.

            r0 and r1 defines the cropping factors along the detector rows. c0 and c1 defines the cropping factors along the detector channels. ::

            :       (0,0)--------------------------(0,1)
            :         |  (r0,c0)---------------+     |
            :         |     |                  |     |
            :         |     | (Cropped Region) |     |
            :         |     |                  |     |
            :         |     +---------------(r1,c1)  |
            :       (1,0)--------------------------(1,1)

            For example, ``crop_factor=[(0.25,0), (0.75,1)]`` will crop out the middle half of the scan image along the vertical direction.

    **Arguments specific to view subsampling**:

        - view_id_start (int): [Default=0] view id corresponding to the first view.
        - view_angle_start (float): [Default=0.0] view angle in radian corresponding to the first view.
        - view_id_end (int): [Default=None] view id corresponding to the last view. If None, this will be equal to the total number of object scan images in ``obj_scan_path``.
        - subsample_view_factor (int): [Default=1]: view subsample factor. By default no view subsampling will be performed.

            For example, with ``subsample_view_factor=2``, every other view will be loaded.

    Returns:
        6-element tuple containing:

        - **obj_scan** (*ndarray, float*): 3D object scan with shape (num_views, num_det_rows, num_det_channels)

        - **blank_scan** (*ndarray, float*): 3D blank scan with shape (1, num_det_rows, num_det_channels)

        - **dark_scan** (*ndarray, float*): 3D dark scan with shape (1, num_det_rows, num_det_channels)

        - **angles** (*ndarray, double*): 1D view angles array in radians in the interval :math:`[0,2\pi)`.

        - **geo_params**: MBIRCONE format geometric parameters containing the following entries:

            - dist_source_detector: Distance between the X-ray source and the detector in units of :math:`ALU`.
            - magnification: Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).

            - delta_det_channel: Detector channel spacing in :math:`ALU`.
            - delta_det_row: Detector row spacing in :math:`ALU`.
            - det_channel_offset: Distance in :math:`ALU` from center of detector to the source-detector line along a row.
            - det_row_offset: Distance in :math:`ALU` from center of detector.
            - rotation_offset: Distance in :math:`ALU` from source-detector line to axis of rotation in the object space.
            - num_det_channels: Number of detector channels.
            - num_det_rows: Number of detector rows.

        - **defective_pixel_list** (list(tuple)): A list of tuples containing indices of invalid sinogram pixels, with the format (detector_row_idx, detector_channel_idx).
    """
    # MBIR geometry parameter dictionary
    geo_params = dict()

    ############### load NSI parameters from the given config file path
    tag_section_list = [['source', 'Result'],
                        ['reference', 'Result'],
                        ['pitch', 'Object Radiograph'],
                        ['width pixels', 'Detector'],
                        ['height pixels', 'Detector'],
                        ['number', 'Object Radiograph'],
                        ['Rotation range', 'CT Project Configuration'],
                        ['rotate', 'Correction'],
                        ['flipH', 'Correction'],
                        ['flipV', 'Correction'],
                        ['angleStep', 'Object Radiograph'],
                        ['clockwise', 'Processed']
                       ]
    assert(os. path. isfile(config_file_path)), f'Error! NSI config file does not exist. Please check whether {config_file_path} is a valid file.'
    NSI_params = _NSI_read_str_from_config(config_file_path, tag_section_list)

    # coordinate of source
    u_s = np.single(NSI_params[0].split(' ')[-1])

    # coordinate of reference
    coordinate_ref = NSI_params[1].split(' ')
    u_d1 = np.single(coordinate_ref[2])
    v_d1 = np.single(coordinate_ref[0])
    w_d1 = np.single(coordinate_ref[1])

    # detector pixel pitch
    pixel_pitch_det = NSI_params[2].split(' ')
    geo_params['delta_det_channel'] = np.single(pixel_pitch_det[0])
    geo_params['delta_det_row'] = np.single(pixel_pitch_det[1])

    # dimension of radiograph
    geo_params['num_det_channels'] = int(NSI_params[3])
    geo_params['num_det_rows'] = int(NSI_params[4])

    # total number of radiograph scans
    num_acquired_scans = int(NSI_params[5])

    # total angles (usually 360 for 3D data, and (360*number_of_full_rotations) for 4D data
    total_angles = int(NSI_params[6])

    # Radiograph rotation (degree)
    scan_rotate = int(NSI_params[7])
    if (scan_rotate == 180) or (scan_rotate == 0):
        print('scans are in portrait mode!')
    elif (scan_rotate == 270) or (scan_rotate == 90):
        print('scans are in landscape mode!')
        geo_params['num_det_channels'], geo_params['num_det_rows'] = geo_params['num_det_rows'], geo_params['num_det_channels']
    else:
        warnings.warn("Picture mode unknown! Should be either portrait (0 or 180 deg rotation) or landscape (90 or 270 deg rotation). Automatically setting picture mode to portrait.")
        scan_rotate = 180

    # Radiograph horizontal & vertical flip
    if NSI_params[8] == "True":
        flipH = True
    else:
        flipH = False
    if NSI_params[9] == "True":
        flipV = True
    else:
        flipV = False

    # Detector rotation angle step (degree)
    angle_step = np.single(NSI_params[10])

    # Detector rotation direction
    if NSI_params[11] == "True":
        print("clockwise rotation.")
    else:
        print("counter-clockwise rotation.")
        # counter-clockwise rotation
        angle_step = -angle_step
    v_d0 = - v_d1
    w_d0 = - w_d1
    geo_params['rotation_offset'] = 0.0

    ############### Adjust geometry NSI_params according to crop_factor and downsample_factor
    if isinstance(crop_factor[0], (list, tuple)):
        (r0, c0), (r1, c1) = crop_factor
    else:
        r0, c0, r1, c1 = crop_factor

    # Adjust parameters after downsampling
    geo_params['num_det_rows'] = (geo_params['num_det_rows'] // downsample_factor[0])
    geo_params['num_det_channels'] = (geo_params['num_det_channels'] // downsample_factor[1])

    geo_params['delta_det_row'] = geo_params['delta_det_row'] * downsample_factor[0]
    geo_params['delta_det_channel'] = geo_params['delta_det_channel'] * downsample_factor[1]

    # Adjust parameters after cropping
    num_det_rows_shift0 = np.round(geo_params['num_det_rows'] * r0)
    num_det_rows_shift1 = np.round(geo_params['num_det_rows'] * (1 - r1))
    w_d0 = w_d0 + num_det_rows_shift0 * geo_params['delta_det_row']
    geo_params['num_det_rows'] = geo_params['num_det_rows'] - (num_det_rows_shift0 + num_det_rows_shift1)

    num_det_channels_shift0 = np.round(geo_params['num_det_channels'] * c0)
    num_det_channels_shift1 = np.round(geo_params['num_det_channels'] * (1 - c1))
    v_d0 = v_d0 + num_det_channels_shift0 * geo_params['delta_det_channel']
    geo_params['num_det_channels'] = geo_params['num_det_channels'] - (num_det_channels_shift0 + num_det_channels_shift1)

    ############### calculate MBIRCONE NSI_params from NSI NSI_params
    geo_params["dist_source_detector"] = u_d1 - u_s
    geo_params["magnification"] = -geo_params["dist_source_detector"] / u_s

    dist_dv_to_detector_corner_from_detector_center = - geo_params['num_det_channels'] * geo_params['delta_det_channel'] / 2.0
    dist_dw_to_detector_corner_from_detector_center = - geo_params['num_det_rows'] * geo_params['delta_det_row'] / 2.0
    geo_params["det_channel_offset"] = -(v_d0 - dist_dv_to_detector_corner_from_detector_center)
    geo_params["det_row_offset"] = - (w_d0 - dist_dw_to_detector_corner_from_detector_center)
    if geo_params_only:
        return geo_params

    ############### read blank scans and dark scans
    blank_scan = np.expand_dims(_read_scan_img(blank_scan_path), axis=0)
    if dark_scan_path is not None:
        dark_scan = np.expand_dims(_read_scan_img(dark_scan_path), axis=0)
    else:
        dark_scan = np.zeros(blank_scan.shape)

    if view_id_end is None:
        view_id_end = num_acquired_scans
    view_ids = list(range(view_id_start, view_id_end, subsample_view_factor))
    obj_scan = _read_scan_dir(obj_scan_path, view_ids)

    # Load defective pixel information
    if defective_pixel_path is not None:
        tag_section_list = [['Defect', 'Defective Pixels']]
        defective_loc = _NSI_read_str_from_config(defective_pixel_path, tag_section_list)
        defective_pixel_list = np.array([defective_pixel_ind.split()[1::-1] for defective_pixel_ind in defective_loc ]).astype(int)
        defective_pixel_list = list(map(tuple, defective_pixel_list))
    else:
        defective_pixel_list = None


    # flip the scans according to flipH and flipV NSI_params
    if flipV:
        print("Flip scans vertically!")
        obj_scan = np.flip(obj_scan, axis=1)
        blank_scan = np.flip(blank_scan, axis=1)
        dark_scan = np.flip(dark_scan, axis=1)
        # adjust the defective pixel information: vertical flip
        if defective_pixel_list is not None:
            for i in range(len(defective_pixel_list)):
                (r,c) = defective_pixel_list[i]
                defective_pixel_list[i] = (blank_scan.shape[1]-r-1, c)
    if flipH:
        print("Flip scans horizontally!")
        obj_scan = np.flip(obj_scan, axis=2)
        blank_scan = np.flip(blank_scan, axis=2)
        dark_scan = np.flip(dark_scan, axis=2)
        # adjust the defective pixel information: horizontal flip
        if defective_pixel_list is not None:
            for i in range(len(defective_pixel_list)):
                (r,c) = defective_pixel_list[i]
                defective_pixel_list[i] = (r, blank_scan.shape[2]-c-1)

    # rotate the scans according to scan_rotate param
    rot_count = scan_rotate // 90
    for n in range(rot_count):
        obj_scan = np.rot90(obj_scan, 1, axes=(2,1))
        blank_scan = np.rot90(blank_scan, 1, axes=(2,1))
        dark_scan = np.rot90(dark_scan, 1, axes=(2,1))
        # adjust the defective pixel information: rotation (clockwise)
        if defective_pixel_list is not None:
            for i in range(len(defective_pixel_list)):
                (r,c) = defective_pixel_list[i]
                defective_pixel_list[i] = (c, blank_scan.shape[2]-r-1)

    # cropping in pixels
    obj_scan, blank_scan, dark_scan, defective_pixel_list = _crop_scans(obj_scan, blank_scan, dark_scan,
                                                                        crop_factor=crop_factor,
                                                                        defective_pixel_list=defective_pixel_list)

    # downsampling in pixels (block-averaging)
    if downsample_factor[0]*downsample_factor[1] > 1:
        obj_scan, blank_scan, dark_scan, defective_pixel_list = _downsample_scans(obj_scan, blank_scan, dark_scan,
                                                                                  downsample_factor=downsample_factor,
                                                                                  defective_pixel_list=defective_pixel_list)

    # compute projection angles based on angle_step and rotation direction
    view_angle_start_deg = np.rad2deg(view_angle_start)
    angle_step *= subsample_view_factor
    angles = np.deg2rad(np.array([(view_angle_start_deg+n*angle_step) % 360.0 for n in range(len(view_ids))]))

    return obj_scan, blank_scan, dark_scan, angles, geo_params, defective_pixel_list


def transmission_CT_compute_sino(obj_scan, blank_scan, dark_scan, defective_pixel_list=None):
    """Given a set of object scans, blank scan, and dark scan, compute the sinogram data with the steps below:

        1. ``sino = -numpy.log((obj_scan-dark_scan) / (blank_scan-dark_scan))``.
        2. Identify the invalid sinogram entries. The invalid sinogram entries are indentified as the union of defective pixel entries (speicified by ``defective_pixel_list``) and sinogram entries with values of inf or Nan.

    Args:
        obj_scan (ndarray, float): 3D object scan with shape (num_views, num_det_rows, num_det_channels).
        blank_scan (ndarray, float): [Default=None] 3D blank scan with shape (num_blank_scans, num_det_rows, num_det_channels). When num_blank_scans>1, the pixel-wise mean will be used as the blank scan.
        dark_scan (ndarray, float): [Default=None] 3D dark scan with shape (num_dark_scans, num_det_rows, num_det_channels). When num_dark_scans>1, the pixel-wise mean will be used as the dark scan.
        defective_pixel_list (optional, list(tuple)): A list of tuples containing indices of invalid sinogram pixels, with the format (view_idx, row_idx, channel_idx) or (detector_row_idx, detector_channel_idx).
            If None, then the defective pixels will be identified as sino entries with inf or Nan values.
    Returns:
        2-element tuple containing:
        - **sino** (*ndarray, float*): Sinogram data with shape (num_views, num_det_rows, num_det_channels).
        - **defective_pixel_list** (list(tuple)): A list of tuples containing indices of invalid sinogram pixels, with the format (view_idx, row_idx, channel_idx) or (detector_row_idx, detector_channel_idx).

    """
    # take average of multiple blank/dark scans, and expand the dimension to be the same as obj_scan.
    blank_scan = 0 * obj_scan + np.mean(blank_scan, axis=0, keepdims=True)
    dark_scan = 0 * obj_scan + np.mean(dark_scan, axis=0, keepdims=True)

    obj_scan = obj_scan - dark_scan
    blank_scan = blank_scan - dark_scan
    sino = -np.log(obj_scan / blank_scan)

    # set the sino pixels corresponding to the provided defective list to 0.0
    if defective_pixel_list is None:
        defective_pixel_list = []
    else:    # if provided list is not None
        for defective_pixel_idx in defective_pixel_list:
            if len(defective_pixel_idx) == 2:
                (r,c) = defective_pixel_idx
                sino[:,r,c] = 0.0
            elif len(defective_pixel_idx) == 3:
                (v,r,c) = defective_pixel_idx
                sino[v,r,c] = 0.0
            else:
                raise Exception("transmission_CT_compute_sino: index information in defective_pixel_list cannot be parsed.")

    # set NaN sino pixels to 0.0
    nan_pixel_list = list(map(tuple, np.argwhere(np.isnan(sino)) ))
    for (v,r,c) in nan_pixel_list:
        sino[v,r,c] = 0.0

    # set Inf sino pixels to 0.0
    inf_pixel_list = list(map(tuple, np.argwhere(np.isinf(sino)) ))
    for (v,r,c) in inf_pixel_list:
        sino[v,r,c] = 0.0

    # defective_pixel_list = union{input_defective_pixel_list, nan_pixel_list, inf_pixel_list}
    defective_pixel_list = list(set().union(defective_pixel_list,nan_pixel_list,inf_pixel_list))

    return sino, defective_pixel_list


def calc_weights(sino, weight_type, defective_pixel_list=None):
    """ Compute the weights used in MBIR reconstruction.

    Args:
        sino (float, ndarray): Sinogram data with either 3D shape (num_views, num_det_rows, num_det_channels).

        weight_type (string): Type of noise model used for data

                - weight_type = 'unweighted' => return numpy.ones(sino.shape).
                - weight_type = 'transmission' => return numpy.exp(-sino).
                - weight_type = 'transmission_root' => return numpy.exp(-sino/2).
                - weight_type = 'emission' => return 1/(numpy.absolute(sino) + 0.1).
        defective_pixel_list (list(tuple)): A list of tuples containing indices of invalid sinogram pixels, with the format (view_idx, row_idx, channel_idx) or (row_dix, channel_idx).
            The corresponding weights of invalid sinogram entries are set to 0.0.

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

    # set weights corresponding to invalid sino entries to 0.0
    if defective_pixel_list is not None:
        print("calc_weights: Setting sino weights corresponding to defective pixels to 0.0.")
        for defective_pixel_idx in defective_pixel_list:
            if len(defective_pixel_idx) == 2:
                (r,c) = defective_pixel_idx
                weights[:,r,c] = 0.0
            elif len(defective_pixel_idx) == 3:
                (v,r,c) = defective_pixel_idx
                weights[v,r,c] = 0.0
            else:
                raise Exception("calc_weights: index information in defective_pixel_list cannot be parsed.")

    return weights


def calc_background_offset(sino, option=0, edge_width=9):
    """ Given a sinogram, automatically calculate the background offset based on the selected option. Available options are:

        **Option 0**: Calculate the background offset using edge_width pixels along the upper, left, and right edges of a median sinogram view.

    Args:
        sino (float, ndarray): Sinogram data with 3D shape (num_views, num_det_rows, num_det_channels).
        option (int, optional): [Default=0] Option of algorithm used to calculate the background offset.
        edge_width(int, optional): [Default=9] Width of the edge regions in pixels. It must be an odd integer >= 3.
    Returns:
        offset (float): Background offset value.
    """

    # Check validity of edge_width value
    assert(isinstance(edge_width, int)), "edge_width must be an integer!"
    if (edge_width % 2 == 0):
        edge_width = edge_width+1
        warnings.warn(f"edge_width of background regions should be an odd number! Setting edge_width to {edge_width}.")

    if (edge_width < 3):
        warnings.warn("edge_width of background regions should be >= 3! Setting edge_width to 3.")
        edge_width = 3

    _, _, num_det_channels = sino.shape

    # calculate mean sinogram
    sino_median=np.median(sino, axis=0)

    # offset value of the top edge region.
    # Calculated as median([median value of each horizontal line in top edge region])
    median_top = np.median(np.median(sino_median[:edge_width], axis=1))

    # offset value of the left edge region.
    # Calculated as median([median value of each vertical line in left edge region])
    median_left = np.median(np.median(sino_median[:, :edge_width], axis=0))

    # offset value of the right edge region.
    # Calculated as median([median value of each vertical line in right edge region])
    median_right = np.median(np.median(sino_median[:, num_det_channels-edge_width:], axis=0))

    # offset = median of three offset values from top, left, right edge regions.
    offset = np.median([median_top, median_left, median_right])
    return offset

def calc_weights_mar(sino, angles, dist_source_detector, magnification,
                     init_recon, metal_threshold,
                     beta=2.0, gamma=4.0,
                     defective_pixel_list=None,
                     delta_det_channel=1.0, delta_det_row=1.0, delta_pixel_image=None,
                     det_channel_offset=0.0, det_row_offset=0.0, rotation_offset=0.0, image_slice_offset=0.0,
                     num_threads=None, verbose=0, lib_path=__lib_path):
    """ Compute the weights used for reducing metal artifacts in MBIR reconstruction. For more information please refer to the `[theory] <theory.html>`_ section in readthedocs.

    Required arguments:
        - **sino** (*ndarray*): Sinogram data with 3D shape (num_det_rows, num_det_channels).
        - **angles** (*ndarray*): 1D array of view angles in radians.
        - **dist_source_detector** (*float*): Distance between the X-ray source and the detector in units of :math:`ALU`.
        - **magnification** (*float*): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
        - **init_recon** (*ndarray*): Initial reconstruction used to identify metal voxels.
        - **metal_threshold** (*float*): Threshold value in units of :math:`ALU^{-1}` used to identify metal voxels. Any voxels in ``init_recon`` with an attenuation coefficient larger than ``metal_threshold`` will be identified as a metal voxel.

    Optional arguments specific to MAR data weights:
        - **beta** (*float, optional*): [Default=2.0] Scalar value in range :math:`>0`.

            A larger ``beta`` improves the noise uniformity, but too large a value may increase the overall noise level.
        - **gamma** (*float, optional*): [Default=4.0] Scalar value in range :math:`>1`.

            A larger ``gamma`` reduces the weight of sinogram entries with metal, but too large a value may reduce image quality inside the metal regions.
        - **defective_pixel_list** (optional, list(tuple)): [Default=None] A list of tuples containing indices of invalid sinogram pixels, with the format (view_idx, row_idx, channel_idx).

            weights=0.0 for invalid sinogram entries.

    Optional arguments inherited from ``cone3D.project``:
        - **delta_det_channel** (*float, optional*): [Default=1.0] Detector channel spacing in :math:`ALU`.
        - **delta_det_row** (*float, optional*): [Default=1.0] Detector row spacing in :math:`ALU`.
        - **delta_pixel_image** (*float, optional*): [Default=None] Image pixel spacing in :math:`ALU`.

            If None, automatically set to ``delta_pixel_detector/magnification``.
        - **det_channel_offset** (*float, optional*): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a row.
        - **det_row_offset** (*float, optional*): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a column.
        - **rotation_offset** (*float, optional*): [Default=0.0] Distance in :math:`ALU` from source-detector line to axis of rotation in the object space.

            This is normally set to zero.

        - **image_slice_offset** (*float, optional*): [Default=0.0] Vertical offset of the image in units of :math:`ALU`.
        - **num_threads** (*int, optional*): [Default=None] Number of compute threads requested when executed.

            If None, ``num_threads`` is set to the number of cores in the system.

        - **verbose** (*int, optional*): [Default=1] Possible values are {0,1,2}, where 0 is quiet, 1 prints minimal reconstruction progress information, and 2 prints the full information.
        - **lib_path** (*str, optional*): [Default=~/.cache/mbircone] Path to directory containing library of forward projection matrices.

    Returns:
        (ndarray): Weights used in mbircone reconstruction, with the same array shape as ``sino``.
    """
    _, num_det_rows, num_det_channels = sino.shape
    metal_mask = np.array(init_recon > metal_threshold, dtype=np.float32)
    metal_mask_projected = cone3D.project(metal_mask, angles,
                                          num_det_rows, num_det_channels,
                                          dist_source_detector, magnification,
                                          delta_det_channel=delta_det_channel, delta_det_row=delta_det_row, delta_pixel_image=delta_pixel_image,
                                          det_channel_offset=det_channel_offset, det_row_offset=det_row_offset, rotation_offset=rotation_offset, image_slice_offset=image_slice_offset,
                                          num_threads=num_threads, verbose=verbose, lib_path=lib_path) 
    sino_mask = metal_mask_projected > 0.0
    weights = np.zeros(sino.shape)
    # weights for sino entries where the projection path does not contain metal voxels
    weights[~sino_mask] = np.exp(-sino[~sino_mask]/beta)
    # weights for sino entries where the projection path contains metal voxels
    weights[sino_mask] = np.exp(-sino[sino_mask]*gamma/beta)
    # weights for invalid sino entries
    if defective_pixel_list is not None:
        print("calc_weights_mar: Setting sino weights corresponding to defective pixels to 0.0.")
        for defective_pixel_idx in defective_pixel_list:
            if len(defective_pixel_idx) == 2:
                (r,c) = defective_pixel_idx
                weights[:,r,c] = 0.0
            elif len(defective_pixel_idx) == 3:
                (v,r,c) = defective_pixel_idx
                weights[v,r,c] = 0.0
            else:
                raise Exception("calc_weights_mar: index information in defective_pixel_list cannot be parsed.")

    return weights
