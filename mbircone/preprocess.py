import os
from glob import glob
import numpy as np
from PIL import Image
import warnings
import math
from scipy.ndimage import convolve
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
        ndarray (float): 3D numpy array, (num_views, num_slices, num_channels). A stack of scan images.
    """

    if view_ids == []:
        warnings.warn("view_ids should not be empty.")

    img_path_list = sorted(glob(os.path.join(scan_dir, '*')))
    img_path_list = [img_path_list[i] for i in view_ids]
    img_list = [_read_scan_img(img_path) for img_path in img_path_list]

    # return shape = num_views x num_slices x num_channels
    return np.stack(img_list, axis=0)


def _downsample_scans(obj_scan, blank_scan, dark_scan, downsample_factor=[1, 1]):
    """Performs Down-sampling to the scan images in the detector plane.

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
    blank_scan = blank_scan.reshape(blank_scan.shape[0], blank_scan.shape[1] // downsample_factor[0],
                                    downsample_factor[0],
                                    blank_scan.shape[2] // downsample_factor[1], downsample_factor[1]).sum((2, 4))
    dark_scan = dark_scan.reshape(dark_scan.shape[0], dark_scan.shape[1] // downsample_factor[0], downsample_factor[0],
                                  dark_scan.shape[2] // downsample_factor[1], downsample_factor[1]).sum((2, 4))

    return obj_scan, blank_scan, dark_scan


def _crop_scans(obj_scan, blank_scan, dark_scan, crop_factor=[(0, 0), (1, 1)]):
    """Crops given scans with given factor.

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


def _compute_sino_and_weight_mask_from_scans(obj_scan, blank_scan, dark_scan):
    """Computes sinogram data and weights mask base on given object scan, blank scan, and dark scan. The weights mask is used to filter out negative values in the corrected object scan and blank scan. For real CT dataset weights mask should be used when calculating sinogram weights.
    Args:
        obj_scan (ndarray): A stack of sinograms. 3D numpy array, (num_views, num_slices, num_channels).
        blank_scan (ndarray) : A blank scan. 3D numpy array, (num_sampled_scans, num_slices, num_channels).
        dark_scan (ndarray):  A dark scan. 3D numpy array, (num_sampled_scans, num_slices, num_channels).
    Returns:
        A tuple (sino, weight_mask) containing:
        - **sino** (*ndarray*): Preprocessed sinogram with shape (num_views, num_slices, num_channels).
        - **weight_mask** (*ndarray*): A binary mask for sinogram weights. 

    """
    blank_scan_mean = 0 * obj_scan + np.mean(blank_scan, axis=0, keepdims=True)
    dark_scan_mean = 0 * obj_scan + np.mean(dark_scan, axis=0, keepdims=True)

    obj_scan_corrected = (obj_scan - dark_scan_mean)
    blank_scan_corrected = (blank_scan_mean - dark_scan_mean)
    sino = -np.log(obj_scan_corrected / blank_scan_corrected)

    weight_mask = (obj_scan_corrected > 0) & (blank_scan_corrected > 0)

    return sino, weight_mask


def _compute_views_index_list(scan_range, num_sampled_scans):
    """Returns a list of sampled indices of views to use for reconstruction.

    Args:
        scan_range ([int, int]): Start and end index corresponding to the sampled scans.
        num_sampled_scans (int): Number of scans to be picked out of total number of scans.

    Returns:
        list[int], a list of sampled view indices.

    """
    index_original = range(scan_range[0], scan_range[1])
    assert num_sampled_scans <= len(index_original), 'num_sampled_scans cannot exceed range of view index'
    index_sampled = [scan_range[0] + int(np.floor(i * len(index_original) / num_sampled_scans)) for i in range(num_sampled_scans)]
    return index_sampled


def _select_contiguous_subset(indexList, num_time_points=1, time_point=0):
    """Returns a contiguous subset of index list corresponding to a given time point. This is mainly used for 4D datasets with multiple time points.

    Args:
        indexList (list[int]): A list of view indices.
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


def _compute_angles_list(view_index_list, num_acquired_scans, total_angles, rotation_direction="positive"):
    """Returns angles list from index list.

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


def NSI_read_params(config_file_path):
    """Reads NSI system parameters from a NSI configuration file.

    Args:
        config_file_path (string): Path to NSI configuration file. The filename extension is '.nsipro'.
    Returns:
        Dictionary: NSI system parameters.
    """
    tag_section_list = [['source', 'Result'],
                        ['reference', 'Result'],
                        ['pitch', 'Object Radiograph'],
                        ['width pixels', 'Detector'],
                        ['height pixels', 'Detector'],
                        ['number', 'Object Radiograph'],
                        ['Rotation range', 'CT Project Configuration']]
    params = _NSI_read_str_from_config(config_file_path, tag_section_list)
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


def NSI_adjust_sysparam(NSI_system_params, downsample_factor=[1, 1], crop_factor=[(0, 0), (1, 1)]):
    """Returns adjusted NSI system parameters given downsampling factor and cropping factor.

    Args:
        NSI_system_params (dict of string-int): NSI system parameters.
        downsample_factor ([int, int]): [Default=[1,1]] Two numbers to define down-sample factor.
        crop_factor ([(int, int),(int, int)] or [int, int, int, int]): [Default=[(0, 0), (1, 1)]]
            Two points to define the bounding box. Sequence of [(r0, c0), (r1, c1)] or [r0, c0, r1, c1], where 1>=r1 >= r0>=0 and 1>=c1 >= c0>=0.
    Returns:
        Dictionary: Adjusted NSI system parameters.
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


def NSI_to_MBIRCONE_params(NSI_system_params):
    """Returns MBIRCONE format geometric parameters from adjusted NSI system parameters.

    Args:
        NSI_system_params (dict of string-int): Adjusted NSI system parameters.
    Returns:
        Dictionary: MBIRCONE format geometric parameters

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


def _gauss2D(window_size=(15, 15)):
    m, n = [(ss - 1.) / 2. for ss in window_size]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    sigma_h = 1.
    h = np.exp(-(x * x + y * y) / (2. * sigma_h * sigma_h))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    w_0 = np.hamming(window_size[0])
    w_1 = np.hamming(window_size[1])
    w = w_0 * w_1
    h = np.multiply(h, w)
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def _image_indicator(image, background_ratio):
    indicator = np.int8(image > np.percentile(image, background_ratio * 100))  # for excluding empty space from average
    return indicator


def _image_mask(image, blur_filter, background_ratio, boundary_ratio):
    ''' Automatic image segmentation:
        1. Blur the input image with a 2D Gaussian filter (with hamming window).
        2. Compute a binary mask that indicates the region of image support.
        3. Set region between ROI and ROR to be 0.

    Args:
        blur_filter (ndarray): blurring filter used to smooth the input image.
        background_ratio (float): Should be a number in [0,1]. This is the estimated ratio of background pixels. For example, background_ratio=0.5 means that 50% of pixels will be recognized as background.
        boundary_ratio (float): Should be a number in [0,1]. This is the estimated ratio of (ROR_radius-ROI_radius)/ROR_radius. The region between ROR and ROI will be recognized as background.           
    Returns:
        ndarray: Masked image with same shape of input image. 
    '''
    # blur the input image with a 2D Gaussian window
    (num_slices, num_rows_cols, _) = np.shape(image)
    image_blurred = np.array([convolve(image[i], blur_filter, mode='wrap') for i in range(num_slices)])
    image_indicator = _image_indicator(image_blurred, background_ratio)
    boundary_len = num_rows_cols * boundary_ratio // 2
    R = (num_rows_cols - 1) * (1 - boundary_ratio) // 2
    center_pt = (num_rows_cols - 1) // 2
    boundary_mask = np.zeros((num_rows_cols, num_rows_cols))
    for i in range(num_rows_cols):
        for j in range(num_rows_cols):
            boundary_mask[i, j] = (
                        np.sum((i - center_pt) * (i - center_pt) + (j - center_pt) * (j - center_pt)) < R * R)
    image_indicator = image_indicator * np.array([np.int8(boundary_mask) for _ in range(num_slices)])
    return image_indicator * image


def background_offset_calibration(sino, background_view_list, background_box_info_list):
    """Performs background offset calibration to the sinogram. 
    
    Args:
        sino (ndarray): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels)
        background_view_list ([int]): [Default=[]] A list of view indices used for background offset calibration. `background_view_list` indicates the views corresponding to the rectangular areas specified in `background_box_info_list`. It should have the same length as `background_box_info_list`.
        background_box_info_list ([(x,y,width,height)]): [default=[]] A list of tuples indicating the information of the rectangular areas used for background offset calculation.  It should have the same length as `background_view_list`.
            For each tuple in `background_box_info_list` it should have the format `(x,y,width,height)`, where `(x,y) represents the anchor point of the rectangle, and `width`and `height` represents the width and height of the Rectangle.    
    Returns:
        float: The background offset calculated from `sino`.
    """
    avg_offset = 0.
    if not background_view_list:
        return 0.
    for view_idx, box_info in zip(background_view_list, background_box_info_list):
        print(f"box used in view {view_idx} for calbiration: (x,y,width,height)=", box_info)
        (x, y, box_width, box_height) = box_info
        avg_offset += np.mean(sino[view_idx, y:y + box_height, x:x + box_width])
    avg_offset /= len(background_view_list)
    return avg_offset


def NSI_process_raw_scans(path_radiographs, NSI_system_params,
                          path_blank=None, path_dark=None,
                          num_time_points=None, time_point=0,
                          num_sampled_scans=None,                      
                          rotation_direction="positive"):
    """Reads a subset of scan images corresponding to the given time point from an NSI ConeBeam scan directory, and calculates view angles corresponding to the object scans.
    
    Args:
        path_radiographs (string): Path to an NSI ConeBeam scan directory.
        NSI_system_params (dict): A dictionary containing NSI parameters. This can be obtained from an NSI configuration file using function `preprocess.NSI_read_params()`.
        num_sampled_scans (int): [Default=None] Number of object scans sampled from all object scans. It should be smaller than the total number of object scans from directory. 
            By default, num_sampled_scans will be the total number of scans in the directory.
            The subset of the scans will be picked by a grid subsampling strategy. For example, by setting num_sampled_scans to be half of total number of object scans in the directory, the algorithm will pick every other object scan. 
        path_blank (string): [Default=None] Path to a blank scan image, e.g. 'path_to_scan/gain0.tif'
        path_dark (string): [Default=None] Path to a dark scan image, e.g. 'path_to_scan/offset.tif'
        rotation_direction (string): [Default='positive'] Rotation direction for angles calculation.
            Should be one of 'positive' or 'negative'.
        num_time_points (int): [Default=None] Total number of time points for all object scans.
            `num_time_points` will be used to partition the sampled object scans into subsets corresponding to different time points.
            By default, the scans will be partitioned such that a time point contains sampled scans in a 360-degree rotation.
        time_point (int): [Default=0] Index of the time point of the 3D object scan we would like to retrieve. 
            `time_point` shoud be in range of [0, num_time_points-1], where 0 corresponds to the first time point.    
    Returns: 
        4-element tuple containing:
        
        - **obj_scan** (*ndarray, float*): 3D object scan with shape (num_views, num_det_rows, num_det_channels)
        
        - **blank_scan** (*ndarray, float*): 3D blank scan with shape (1, num_det_rows, num_det_channels)
        
        - **dark_scan** (*ndarray, float*): 3D dark scan with shape (1, num_det_rows, num_det_channels)
        
        - **angles** (*ndarray, double*): 1D array of view angles in radians. 'angles[k]' is the angle for view :math:`k` of object scan. It is assumed that the rotation of each view is equally spaced in range :math:`[0,2\pi)`.
    """

    num_acquired_scans = NSI_system_params["num_acquired_scans"]
    total_angles = NSI_system_params["total_angles"]
    if num_sampled_scans is None:
        num_sampled_scans = num_acquired_scans
    if num_time_points is None:
        num_time_points = total_angles//360 
    scan_ids = _compute_views_index_list([0, num_acquired_scans], num_sampled_scans)
    view_ids = _select_contiguous_subset(scan_ids, num_time_points, time_point)
     
    angles = _compute_angles_list(view_ids, num_acquired_scans, total_angles, rotation_direction)
    obj_scan = _read_scan_dir(path_radiographs, view_ids)
    print("raw obj scan max value = ", np.max(obj_scan))
    print("raw obj scan min value = ", np.min(obj_scan))

    # Should deal with situation when input is None.
    if path_blank is not None:
        blank_scan = np.expand_dims(_read_scan_img(path_blank), axis=0)
    else:
        blank_scan = np.expand_dims(0 * obj_scan[0] + 1, axis=0)
    if path_dark is not None:
        dark_scan = np.expand_dims(_read_scan_img(path_dark), axis=0)
    else:
        dark_scan = np.expand_dims(0 * obj_scan[0], axis=0)

    obj_scan = np.flip(obj_scan, axis=1)
    blank_scan = np.flip(blank_scan, axis=1)
    dark_scan = np.flip(dark_scan, axis=1)
    return obj_scan, blank_scan, dark_scan, angles


def compute_sino_from_scans(obj_scan, blank_scan=None, dark_scan=None,
                            downsample_factor=[1, 1], crop_factor=[(0, 0), (1, 1)],
                            weight_type='unweighted',
                            background_view_list=[], background_box_info_list=[]):
    """Given a set of object scan, blank scan, and dark scan, compute the sinogram used for reconstruction. This function will downsample and crop the scans, compute sinogram and weights from the scans, and finally perform background offset calibration to the sinogram.

    Args:
        obj_scan (ndarray, float): 3D object scan with shape (num_views, num_det_rows, num_det_channels).
        blank_scan (ndarray, float): [Default=None] 3D blank scan with shape (num_blank_scans, num_det_rows, num_det_channels).
        dark_scan (ndarray, float): [Default=None] 3D dark scan with shape (num_dark_scans, num_det_rows, num_det_channels)
        downsample_factor ([int, int]): [Default=[1,1]] Two numbers to define down-sample factor.
        crop_factor ([(float, float),(float, float)] or [float, float, float, float]): [Default=[(0., 0.), (1., 1.)]]
            Two fractional points to define the bounding box. Sequence of [(r0, c0), (r1, c1)] or [r0, c0, r1, c1], where 1>=r1 >= r0>=0 and 1>=c1 >= c0>=0.
        weight_type (string, optional): [Default='unweighted'] Type of noise model used for data.
            The function ``cone3D.calc_weights`` is used to set weights using specified ``weight_type`` parameter.
                - Option "unweighted" corresponds to unweighted reconstruction;
                - Option "transmission" is the correct weighting for transmission CT with constant dosage;
                - Option "transmission_root" is commonly used with transmission CT data to improve image homogeneity;
                - Option "emission" is appropriate for emission CT data.
        background_view_list ([int]): [Default=[]] A list of view indices used for background offset calibration. `background_view_list` indicates the views corresponding to the rectangular areas specified in `background_box_info_list`. It should have the same length as `background_box_info_list`.
        background_box_info_list ([(x,y,width,height)]): [default=[]] A list of tuples indicating the information of the rectangular areas used for background offset calculation.  It should have the same length as `background_view_list`.
            For each tuple in `background_box_info_list` it should have the format `(x,y,width,height)`, where `(x,y) represents the anchor point of the rectangle, and `width`and `height` represents the width and height of the Rectangle. 
    Returns:
        2-element tuple containing:
        
        - **sino** (*ndarray, float*): Preprocessed 3D sinogram.
        
        - **weights** (*ndarray, float*): 3D weights array with the same shape as sino. 
    
    """

    # set default blank and dark scans if None.
    if blank_scan is None:
        blank_scan = np.expand_dims(0 * obj_scan[0] + 1, axis=0) 
    if dark_scan is None:
        dark_scan = np.expand_dims(0 * obj_scan[0], axis=0)
    # downsampling in pixels
    obj_scan, blank_scan, dark_scan = _downsample_scans(obj_scan, blank_scan, dark_scan,
                                                        downsample_factor=downsample_factor)
    # cropping in pixels
    obj_scan, blank_scan, dark_scan = _crop_scans(obj_scan, blank_scan, dark_scan,
                                                  crop_factor=crop_factor)
    sino, weight_mask = _compute_sino_and_weight_mask_from_scans(obj_scan, blank_scan, dark_scan)

    # set invalid sinogram entries to 0
    sino[weight_mask == 0] = 0.
    # compute sinogram weights
    weights = cone3D.calc_weights(sino, weight_type=weight_type)
    # set the weights corresponding to invalid sinogram entries to 0.
    weights[weight_mask == 0] = 0.
    # background offset calibration
    background_offset = background_offset_calibration(sino, background_view_list, background_box_info_list)
    print("background offset = ", background_offset)
    sino = sino - background_offset
    return sino.astype(np.float32), weights.astype(np.float32)
