import os
from glob import glob
import numpy as np
from PIL import Image
import warnings
import math


def read_scan_img(img_path):
    """Read and return single image from a ConeBeam Scan.

    Args:
        img_path (string): Path to a ConeBeam Scan.
    Returns:
        ndarray (float): 2D numpy array. A single sinogram.
    """

    img = np.asarray(Image.open(img_path))

    if np.issubdtype(img.dtype, np.integer):
        # make float and normalize integer types
        maxval = np.iinfo(img.dtype).max
        img = img.astype(np.float32) / maxval

    return img.astype(np.float32)


def read_scan_dir(scan_dir, view_ids=[]):
    """Read and return a stack of sinograms from a directory.

    Args:
        scan_dir (string): Path to a ConeBeam Scan directory.
        view_ids (list[int]): List of view indexes to specify which scans to read.
    Returns:
        ndarray (float): 3D numpy array, (num_views, num_slices, num_channels). A stack of sinograms.
    """

    if view_ids == []:
        warnings.warn("view_ids should not be empty.")

    img_path_list = sorted(glob(os.path.join(scan_dir, '*')))
    img_path_list = [img_path_list[i] for i in view_ids]
    img_list = [read_scan_img(img_path) for img_path in img_path_list]

    # return shape = num_views x num_slices x num_channels
    return np.stack(img_list, axis=0)


def downsample_scans(obj_scan, blank_scan, dark_scan, downsample_factor=[1, 1]):
    """Down-sample given scans with given factor.

    Args:
        obj_scan (float): A stack of sinograms. 3D numpy array, (num_views, num_slices, num_channels).
        blank_scan (float): A blank scan. 3D numpy array, (1, num_slices, num_channels).
        dark_scan (float): A dark scan. 3D numpy array, (1, num_slices, num_channels).
        downsample_factor ([int, int]): Default=[1,1]] Two numbers to define down-sample factor.
    Returns:
        Downsampled scans

        - **obj_scan** (*ndarray, float*): A stack of sinograms. 3D numpy array, (num_views, num_slices, num_channels).
        - **blank_scan** (*ndarray, float*): A blank scan. 3D numpy array, (1, num_slices, num_channels).
        - **dark_scan** (*ndarray, float*): A dark scan. 3D numpy array, (1, num_slices, num_channels).
    """

    assert len(downsample_factor) == 2, 'factor({}) needs to be of len 2'.format(downsample_factor)

    new_size1 = downsample_factor[0] * (obj_scan.shape[1] // downsample_factor[0])
    new_size2 = downsample_factor[1] * (obj_scan.shape[2] // downsample_factor[1])

    obj_scan = obj_scan[:, 0:new_size1, 0:new_size2]
    blank_scan = blank_scan[:, 0:new_size1, 0:new_size2]
    dark_scan = dark_scan[:, 0:new_size1, 0:new_size2]

    obj_scan = obj_scan.reshape(obj_scan.shape[0], obj_scan.shape[1] // downsample_factor[0], downsample_factor[0],
                                obj_scan.shape[2] // downsample_factor[1], downsample_factor[1]).sum((2, 4))
    blank_scan = blank_scan.reshape(blank_scan.shape[0], blank_scan.shape[1] // downsample_factor[0], downsample_factor[0],
                                    blank_scan.shape[2] // downsample_factor[1], downsample_factor[1]).sum((2, 4))
    dark_scan = dark_scan.reshape(dark_scan.shape[0], dark_scan.shape[1] // downsample_factor[0], downsample_factor[0],
                                  dark_scan.shape[2] // downsample_factor[1], downsample_factor[1]).sum((2, 4))

    return obj_scan, blank_scan, dark_scan


def crop_scans(obj_scan, blank_scan, dark_scan, crop_factor=[(0, 0), (1, 1)]):
    """Crop given scans with given factor.

    Args:
        obj_scan (float): A stack of sinograms. 3D numpy array, (num_views, num_slices, num_channels).
        blank_scan (float) : A blank scan. 3D numpy array, (1, num_slices, num_channels).
        dark_scan (float): A dark scan. 3D numpy array, (1, num_slices, num_channels).
        crop_factor ([(int, int),(int, int)] or [int, int, int, int]):
            [Default=[(0, 0), (1, 1)]] Two points to define the bounding box. Sequence of [(r0, c0), (r1, c1)] or
            [r0, c0, r1, c1], where 1>=r1 >= r0>=0 and 1>=c1 >= c0>=0.

    Returns:
        Cropped scans

        - **obj_scan** (*ndarray, float*): A stack of sinograms. 3D numpy array, (num_views, num_slices, num_channels).
        - **blank_scan** (*ndarray, float*): A blank scan. 3D numpy array, (1, num_slices, num_channels).
        - **dark_scan** (*ndarray, float*): A dark scan. 3D numpy array, (1, num_slices, num_channels).

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

    return obj_scan, blank_scan, dark_scan


def compute_sino(obj_scan, blank_scan, dark_scan):
    """Compute sinogram data base on given object scan, blank scan, and dark scan.

    Args:
        obj_scan (float): A stack of sinograms. 3D numpy array, (num_views, num_slices, num_channels).
        blank_scan (float) : A blank scan. 3D numpy array, (1, num_slices, num_channels).
        dark_scan (float):  A dark scan. 3D numpy array, (1, num_slices, num_channels).
    Returns:
        ndarray (float): Preprocessed sinograms. 3D numpy array, (num_views, num_slices, num_channels).

    """
    blank_scan_mean = 0 * obj_scan + np.average(blank_scan, axis=0)
    dark_scan_mean = 0 * obj_scan + np.average(dark_scan, axis=0)

    obj_scan_corrected = (obj_scan - dark_scan_mean)
    blank_scan_corrected = (blank_scan_mean - dark_scan_mean)

    good_pixels = (obj_scan_corrected > 0) & (blank_scan_corrected > 0)

    normalized_scan = np.zeros(obj_scan_corrected.shape)
    normalized_scan[good_pixels] = obj_scan_corrected[good_pixels] / blank_scan_corrected[good_pixels]

    sino = np.zeros(obj_scan_corrected.shape)
    sino[normalized_scan > 0] = -np.log(normalized_scan[normalized_scan > 0])

    return sino


def compute_views_index_list(view_range, num_views):
    """Return a list of sampled indexes of views to use for reconstruction.

    Args:
        view_range ([int, int]): Two indexes of views to specify the range of views to use for reconstruction.
        num_views (int): Number of views to use for reconstruction.

    Returns:
        list[int], a list of sampled indexes of views to use for reconstruction.

    """
    index_original = range(view_range[0], view_range[1] + 1)
    assert num_views <= len(index_original), 'num_views cannot exceed range of view index'
    index_sampled = [view_range[0] + int(np.floor(i * len(index_original) / num_views)) for i in range(num_views)]
    return index_sampled


def select_contiguous_subset(indexList, num_time_points=1, time_point=0):
    """Return a contiguous subset of index list given a time point. Mainly use for 4D datasets with multiple time points.

    Args:
        indexList (list[int]): A list of indexes.
        num_time_points (int): [Default=1] Total number of time points.
        time_point (int): [Default=0] Index of the time point we want to use for 3D reconstruction.

    Returns:
        list[int], a contiguous subset of index list.
    """
    assert time_point < num_time_points, 'ind_chunk cannot be larger than num_chunk.'

    len_ind = len(indexList)
    num_per_set = len_ind // num_time_points

    # distribute the remaining
    remaining_index_num = len_ind % num_time_points

    start_id = time_point * num_per_set + np.minimum(time_point, remaining_index_num)
    end_id = (time_point + 1) * num_per_set + np.minimum(time_point + 1, remaining_index_num)

    indexList_new = indexList[start_id:end_id]
    return indexList_new


def compute_angles_list(view_index_list, num_acquired_scans, total_angles, rotation_direction="positive"):
    """Return angles list from indexes list.

    Args:
        view_index_list (list[int]): Final index list after subsampling and time point selection.
        num_acquired_scans (int): Total number of acquired scans in the directory.
        total_angles (int): Total rotation angle for the whole dataset.
        rotation_direction (string): [Default='positive'] Rotation direction. Should be 'positive' or 'negative'.

    Returns:
        ndarray (float), 1D view angles array in radians.
    """
    if rotation_direction not in ["positive", "negative"]:
        warnings.warn("Parameter rotation_direction is not valid string; Setting center_offset = 'positive'.")
        rotation_direction = "positive"
    view_index_list = np.array(view_index_list)
    if rotation_direction == "positive":
        angles_list = (total_angles * view_index_list / num_acquired_scans) % 360.0 / 360.0 * (2 * np.pi)
    if rotation_direction == "negative":
        angles_list = (-total_angles * view_index_list / num_acquired_scans) % 360.0 / 360.0 * (2 * np.pi)

    return angles_list


def read_NSI_string(filepath, tags_sections):
    """Return strings about dataset information read from NSI configuration file.

    Args:
        filepath (string): Path to NSI configuration file. The filename extension is '.nsipro'.
        tags_sections (list[string,string]): Given tags and sections to locate the information we want to read.
    Returns:
        list[string], a list of strings have needed dataset information for reconstruction.

    """
    tag_strs = ['<' + tag + '>' for tag, section in tags_sections]
    section_starts = ['<' + section + '>' for tag, section in tags_sections]
    section_ends = ['</' + section + '>' for tag, section in tags_sections]
    params = []

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
                    params.append("")
                else:
                    params.append(line[tag_ind:].strip('\n'))

    return params


def read_NSI_params(filepath):
    """ Read NSI system parameters

    Args:
        filepath (string): Path to NSI configuration file. The filename extension is '.nsipro'.
    Returns:
        dict of string-int: NSI system parameters.
    """
    tag_section_list = [['source', 'Result'],
                        ['reference', 'Result'],
                        ['pitch', 'Object Radiograph'],
                        ['width pixels', 'Detector'],
                        ['height pixels', 'Detector'],
                        ['number', 'Object Radiograph'],
                        ['Rotation range', 'CT Project Configuration']]
    params = read_NSI_string(filepath, tag_section_list)
    NSI_system_params = dict()

    NSI_system_params['u_s'] = np.double(params[0].split(' ')[-1])

    vals = params[1].split(' ')
    NSI_system_params['u_d1'] = np.double(vals[2])
    NSI_system_params['v_d1'] = np.double(vals[0])

    NSI_system_params['w_d1'] = np.double(vals[1])

    vals = params[2].split(' ')
    NSI_system_params['delta_dv'] = np.double(vals[0])
    NSI_system_params['delta_dw'] = np.double(vals[1])

    NSI_system_params['N_dv'] = int(params[3])
    NSI_system_params['N_dw'] = int(params[4])
    NSI_system_params['num_acquired_scans'] = int(params[5])
    NSI_system_params['total_angles'] = int(params[6])

    NSI_system_params['v_d0'] = - NSI_system_params['v_d1']
    NSI_system_params['w_d0'] = - NSI_system_params['N_dw'] * NSI_system_params['delta_dw'] / 2.0
    NSI_system_params['v_r'] = 0.0
    return NSI_system_params


def adjust_NSI_sysparam(NSI_system_params, downsample_factor=[1, 1], crop_factor=[(0, 0), (1, 1)]):
    """Return adjusted NSI system parameters given downsampling factor and cropping factor.

    Args:
        NSI_system_params (dict of string-int): NSI system parameters.
        downsample_factor ([int, int]): [Default=[1,1]] Two numbers to define down-sample factor.
        crop_factor ([(int, int),(int, int)] or [int, int, int, int]): [Default=[(0, 0), (1, 1)]]
            Two points to define the bounding box. Sequence of [(r0, c0), (r1, c1)] or [r0, c0, r1, c1], where 1>=r1 >= r0>=0 and 1>=c1 >= c0>=0.
    Returns:
        dict of string-int: Adjusted NSI system parameters.
    """
    if isinstance(crop_factor[0], (list, tuple)):
        (r0, c0), (r1, c1) = crop_factor
    else:
        r0, c0, r1, c1 = crop_factor

    # Adjust parameters after downsampling
    NSI_system_params['N_dw'] = (NSI_system_params['N_dw'] // downsample_factor[0])
    NSI_system_params['N_dv'] = (NSI_system_params['N_dv'] // downsample_factor[1])

    NSI_system_params['delta_dw'] = NSI_system_params['delta_dw'] * downsample_factor[0]
    NSI_system_params['delta_dv'] = NSI_system_params['delta_dv'] * downsample_factor[1]

    # Adjust parameters after cropping

    N_dwshift0 = np.round(NSI_system_params['N_dw'] * r0)
    N_dwshift1 = np.round(NSI_system_params['N_dw'] * (1 - r1))
    NSI_system_params['w_d0'] = NSI_system_params['w_d0'] + N_dwshift0 * NSI_system_params['delta_dw']
    NSI_system_params['N_dw'] = NSI_system_params['N_dw'] - (N_dwshift0 + N_dwshift1)

    N_dvshift0 = np.round(NSI_system_params['N_dv'] * c0)
    N_dvshift1 = np.round(NSI_system_params['N_dv'] * (1 - c1))
    NSI_system_params['v_d0'] = NSI_system_params['v_d0'] + N_dvshift0 * NSI_system_params['delta_dv']
    NSI_system_params['N_dv'] = NSI_system_params['N_dv'] - (N_dvshift0 + N_dvshift1)

    return NSI_system_params


def transfer_NSI_to_MBIRCONE(NSI_system_params):
    """Return MBIRCONE format geometric parameters from adjusted NSI system parameters.

    Args:
        NSI_system_params (dict of string-int): Adjusted NSI system parameters.
    Returns:
        dict of string-int: MBIRCONE format geometric parameters

    """
    geo_params = dict()
    geo_params["num_channels"] = NSI_system_params['N_dv']
    geo_params["num_slices"] = NSI_system_params['N_dw']
    geo_params["delta_pixel_detector"] = NSI_system_params['delta_dv']
    geo_params["rotation_offset"] = NSI_system_params['v_r']

    geo_params["dist_source_detector"] = NSI_system_params['u_d1'] - NSI_system_params['u_s']
    geo_params["magnification"] = -geo_params["dist_source_detector"] / NSI_system_params['u_s']

    dist_dv_to_detector_corner_from_detector_center = - NSI_system_params['N_dv'] * NSI_system_params['delta_dw'] / 2.0
    dist_dw_to_detector_corner_from_detector_center = - NSI_system_params['N_dw'] * NSI_system_params['delta_dv'] / 2.0
    geo_params["channel_offset"] = -(NSI_system_params['v_d0'] - dist_dv_to_detector_corner_from_detector_center)
    geo_params["row_offset"] = - (NSI_system_params['w_d0'] - dist_dw_to_detector_corner_from_detector_center)
    return geo_params


def preprocess(path_radiographs, num_views, path_blank=None, path_dark=None,
               view_range=None, total_angles=360, num_acquired_scans=2000,
               rotation_direction="positive", downsample_factor=[1, 1], crop_factor=[(0, 0), (1, 1)],
               num_time_points=1, time_point=0):
    """Return preprocessed sinogram and angles list use for reconstruction.

    Args:
        path_radiographs (string): Path to a ConeBeam Scan directory.
        num_views (int): Number of views to use for reconstruction.
        path_blank (string): [Default=None] Path to blank scan.
        path_dark (string): [Default=None] Path to dark scan.
        view_range (list[int, int]): [Default=None] Two indexes of views to specify the range of views to use for reconstruction.
        total_angles (int): [Default=360] Total rotation angle for the whole dataset.
        num_acquired_scans (int): [Default=2000] Total number of acquired scans in the directory.
        rotation_direction (string): [Default='positive'] Rotation direction. Should be 'positive' or 'negative'.
        downsample_factor ([int, int]): [Default=[1,1]] Two numbers to define down-sample factor.
        crop_factor ([(int, int),(int, int)] or [int, int, int, int]): [Default=[(0, 0), (1, 1)]]
            Two points to define the bounding box. Sequence of [(r0, c0), (r1, c1)] or [r0, c0, r1, c1], where 1>=r1 >= r0>=0 and 1>=c1 >= c0>=0.
        num_time_points (int): [Default=1] Total number of time points.
        time_point (int): [Default=0] Index of the time point we want to use for 3D reconstruction.
    Returns:
        2-element tuple containing

        - **sino** (*ndarray, float*): Preprocessed 3D sinogram.

        - **angles** (*array, double*): 1D array of angles corresponding to preprocessed sinogram.

    """

    if view_range is None:
        view_range = [0, num_acquired_scans - 1]

    view_ids = compute_views_index_list(view_range, num_views)
    view_ids = select_contiguous_subset(view_ids, num_time_points, time_point)
    angles = compute_angles_list(view_ids, num_acquired_scans, total_angles, rotation_direction)
    obj_scan = read_scan_dir(path_radiographs, view_ids)

    # Should deal with situation when input is None.
    if path_blank is not None:
        blank_scan = np.expand_dims(read_scan_img(path_blank), axis=0)
    else:
        blank_scan = 0 * obj_scan[0] + 1
        blank_scan = blank_scan.reshape([1, obj_scan.shape[1], obj_scan.shape[2]])
    if path_dark is not None:
        dark_scan = np.expand_dims(read_scan_img(path_dark), axis=0)
    else:
        dark_scan = 0 * obj_scan[0]
        dark_scan = dark_scan.reshape([1, obj_scan.shape[1], obj_scan.shape[2]])

    obj_scan = np.flip(obj_scan, axis=1)
    blank_scan = np.flip(blank_scan, axis=1)
    dark_scan = np.flip(dark_scan, axis=1)

    # downsampling in pixels
    obj_scan, blank_scan, dark_scan = downsample_scans(obj_scan, blank_scan, dark_scan,
                                                       downsample_factor=downsample_factor)
    # cropping in pixels
    obj_scan, blank_scan, dark_scan = crop_scans(obj_scan, blank_scan, dark_scan,
                                                 crop_factor=crop_factor)

    sino = compute_sino(obj_scan, blank_scan, dark_scan)
    return sino.astype(np.float32), angles.astype(np.float64)
