import os
from glob import glob
import numpy as np
from PIL import Image
import warnings
import math
from scipy.ndimage import convolve
from mbircone import cone3D
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

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
        blank_scan (float): A blank scan. 2D numpy array, (num_slices, num_channels).
        dark_scan (float): A dark scan. 3D numpy array, (num_slices, num_channels).
        downsample_factor ([int, int]): Default=[1,1]] Two numbers to define down-sample factor.
    Returns:
        Downsampled scans
        - **obj_scan** (*ndarray, float*): A stack of sinograms. 3D numpy array, (num_views, num_slices, num_channels).
        - **blank_scan** (*ndarray, float*): A blank scan. 3D numpy array, (num_slices, num_channels).
        - **dark_scan** (*ndarray, float*): A dark scan. 3D numpy array, (num_slices, num_channels).
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
            [r0, c0, r1, c1], where 0<=r0 <= r1<=1 and 0<=c0 <= c1<=1.

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
    print('Set sinogram weight corresponding to nan and inf pixels to 0.')
    weight_mask[np.isnan(sino)] = False
    weight_mask[np.isinf(sino)] = False
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


def NSI_read_defect_pixel_loc(defect_file_path):
    tag_section_list = [['Defect', 'Defective Pixels']]
    defect_loc = _NSI_read_str_from_config(defect_file_path, tag_section_list)
    defect_loc = np.array([defect_pixel_ind.split()[:2] for defect_pixel_ind in defect_loc ]).astype(int)
    return defect_loc 

def NSI_read_params(config_file_path, flip_d0=(0, 0.5), transpose=False):
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

    NSI_system_params['u_s'] = np.single(params[0].split(' ')[-1])

    vals = params[1].split(' ')
    NSI_system_params['u_d1'] = np.single(vals[2])
    NSI_system_params['v_d1'] = np.single(vals[0])

    NSI_system_params['w_d1'] = np.single(vals[1])

    vals = params[2].split(' ')
    NSI_system_params['delta_dv'] = np.single(vals[0])
    NSI_system_params['delta_dw'] = np.single(vals[1])

    NSI_system_params['N_dv'] = int(params[3])
    NSI_system_params['N_dw'] = int(params[4])

    NSI_system_params['num_acquired_scans'] = int(params[5])
    NSI_system_params['total_angles'] = int(params[6])
    if transpose:
        NSI_system_params['N_dv'], NSI_system_params['N_dw'] = NSI_system_params['N_dw'], NSI_system_params['N_dv']
    (flip_v_d0, flip_w_d0) = flip_d0
    
    if flip_v_d0 == 0:
        NSI_system_params['v_d0'] = - NSI_system_params['v_d1']
    elif flip_v_d0 == 1:
        NSI_system_params['v_d0'] = NSI_system_params['v_d1'] - NSI_system_params['N_dv'] * NSI_system_params['delta_dv']
    elif flip_v_d0 == 0.5:
        NSI_system_params['v_d0'] = - NSI_system_params['N_dv'] * NSI_system_params['delta_dv'] / 2.0
    else:
        raise ValueError("Unknown flip_v_d0 value. Must be one of 0, 0.5, 1")
    
    if flip_w_d0 == 0:
        NSI_system_params['w_d0'] = - NSI_system_params['w_d1']
    elif flip_w_d0 == 1:
        NSI_system_params['w_d0'] = NSI_system_params['w_d1'] - NSI_system_params['N_dw'] * NSI_system_params['delta_dw']
    elif flip_w_d0 == 0.5:
        NSI_system_params['w_d0'] = - NSI_system_params['N_dw'] * NSI_system_params['delta_dw'] / 2.0
    else:
        raise ValueError("Unknown flip_w_d0 value. Must be one of 0, 0.5, 1")

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


def calc_cone_angle(detector_width, dist_source_detector):
    """ Calculate coneangle along detector rows.      
    """
    return 2*np.arctan(detector_width/2./dist_source_detector) 


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

    dist_dv_to_detector_corner_from_detector_center = - NSI_system_params['N_dv'] * NSI_system_params['delta_dv'] / 2.0
    dist_dw_to_detector_corner_from_detector_center = - NSI_system_params['N_dw'] * NSI_system_params['delta_dw'] / 2.0
    geo_params["det_channel_offset"] = -(NSI_system_params['v_d0'] - dist_dv_to_detector_corner_from_detector_center)
    geo_params["det_row_offset"] = - (NSI_system_params['w_d0'] - dist_dw_to_detector_corner_from_detector_center)
    return geo_params


def _image_indicator(image, background_ratio):
    indicator = np.int8(image > np.percentile(image, background_ratio * 100))  # for excluding empty space from average
    return indicator


def _circle_points(center, radius):
    """Generates points which define a circle on an image.Centre refers to the centre of the circle
    Args:
        center ([int,int]): center coordinate of the circle.
        radius (float): radius of the circle 
    """   
    resolution = round(2*np.pi*radius)
    radians = np.linspace(0, 2*np.pi, resolution)
    c = center[1] + radius*np.cos(radians)#polar co-ordinates
    r = center[0] + radius*np.sin(radians)
    
    return np.array([c, r]).T


def _image_mask_old(image, background_ratio, boundary_ratio, blur_filter=None):
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
    (num_slices, num_rows_cols, _)  = np.shape(image)
    if not (blur_filter is None):
        image = np.array([convolve(image[i], blur_filter, mode='wrap') for i in range(num_slices)])
    image_indicator = _image_indicator(image, background_ratio)
    boundary_len = num_rows_cols*boundary_ratio//2
    R = (num_rows_cols-1)*(1-boundary_ratio)//2
    center_pt = (num_rows_cols-1)//2
    boundary_mask = np.zeros((num_rows_cols, num_rows_cols))
    for i in range(num_rows_cols):
        for j in range(num_rows_cols):
            boundary_mask[i,j] = (np.sum((i-center_pt)*(i-center_pt) + (j-center_pt)*(j-center_pt)) < R*R)
    image_indicator = image_indicator * np.array([np.int8(boundary_mask) for _ in range(num_slices)])
    return image_indicator*image


def generate_2D_mask_from_snake(snake, num_rows_cols):
    mask = np.zeros((num_rows_cols, num_rows_cols))
    i_min = round(np.min(snake[:,0]))
    i_max = round(np.max(snake[:,0]))
    j_min = round(np.min(snake[:,1]))
    j_max = round(np.max(snake[:,1]))
    snake_polypoints = [(snake[i,0],snake[i,1]) for i in range(snake.shape[0])]
    snake_polygon = Polygon(snake_polypoints)
    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            mask[i,j] = snake_polygon.contains(Point(i,j))
    return mask


def smooth_adjacent_masks(mask):
    mask_smooth = np.copy(mask)
    mask_smooth[0] = mask[0] + mask[1]
    for i in range(1, len(mask)-1): 
        mask_smooth[i] = mask[i-1] + mask[i] + mask[i+1]
    mask_smooth[len(mask)-1] = mask[len(mask)-2] + mask[len(mask)-1]
    return mask_smooth.astype(bool) 


def image_mask(image, roi_ratio, use_active_contour, gauss_sigma, alpha, beta, w_line, w_edge, gamma):
    ''' Automatic image segmentation:
        1. Blur the input image with a 2D Gaussian filter (with hamming window).
        2. Compute a binary mask that indicates the region of image support.
        3. Set region between ROI and ROR to be 0.

    Args:
        roi_ratio (float): Should be a number in [0,1]. This is the ratio of ROI_radius/ROR_radius. The region between ROR and ROI will be marked as background.           
        blur_filter (ndarray): [default=None] blurring filter used to smooth the input image. 
            If None, then the blurring step will be skipped.
    Returns:
        ndarray: Masked image with same shape of input image. 
    '''
    # blur the input image with a 2D Gaussian window
    (num_slices, num_rows_cols, _) = image.shape
    image = np.array([gaussian(image[i], gauss_sigma, preserve_range=True) for i in range(num_slices)])
    roi_radius = num_rows_cols * roi_ratio / 2
    center_pt = num_rows_cols // 2
    roi_limit_points = _circle_points([center_pt, center_pt], roi_radius)
    if use_active_contour:
        print("Use active contour detection algorithm!")
        if alpha is None:
            alpha = 0.00037*num_rows_cols
            print(f"alpha automatically calculated. alpha={alpha:.4f}")
        snake = np.array([active_contour(image[i], roi_limit_points, alpha=alpha, w_line=w_line, w_edge=w_edge, beta=beta, gamma=gamma) for i in range(num_slices)])
        mask = np.array([generate_2D_mask_from_snake(snake[i], num_rows_cols) for i in range(num_slices)])
        mask = smooth_adjacent_masks(mask)
    else:
        print("Use default contour of ROI circle.")
        mask_2D = generate_2D_mask_from_snake(roi_limit_points, num_rows_cols)
        mask = np.array([np.copy(mask_2D) for _ in range(num_slices)])
    return mask


def blind_fixture_correction(sino, angles, dist_source_detector, magnification,
                             recon_init=None, 
                             gauss_sigma=2., roi_ratio=0.9, use_active_contour=True, 
                             alpha=None, beta=10., w_line=-0.5, w_edge=1.5, gamma=0.01, 
                             det_channel_offset=0.0, det_row_offset=0.0, rotation_offset=0.0,
                             delta_pixel_detector=1.0, delta_pixel_image=None, ror_radius=None,
                             init_image=0.0,
                             sigma_y=None, snr_db=30.0, weights=None, weight_type='unweighted',
                             positivity=True, p=1.2, q=2.0, T=1.0, num_neighbors=6,
                             sharpness=0.0, sigma_x=None, max_iterations=20, stop_threshold=0.02,
                             num_threads=None, NHICD=False, verbose=1, lib_path=__lib_path):
    """ Corrects the sinogram data for fixtures placed out of the field of view of the scanner.
    
    Required arguments:
        - **sino** (*ndarray*): 3D sinogram array with shape (num_views, num_det_rows, num_det_channels)
        - **angles** (*ndarray*): 1D view angles array in radians.
        - **dist_source_detector** (*float*): Distance between the X-ray source and the detector in units of ALU
        - **magnification** (*float*): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
    Optional arguments specific to blind fixture correction:
        - **recon_init** (*ndarray, optional*): [Default=None] Reconstruction from input sinogram. It is assumed that ``recon_init`` is the reconstruction corresponding to the input ``sino`` (This is important in order for the algorithm to work as expected). If None, a qGGMRF reconstruction based on ``sino`` will be used as ``recon_init``. 
        - **gauss_sigma** (*float, optional*): [Default=2.] standard deviation of Gaussian filter used in both image masking and sinogram error blurring.
        - **roi_ratio** (*float, optional*): [Default=0.9] Should be a number in [0,1]. This is the ratio of ROI_radius/ROR_radius. The region between ROR and ROI will be marked as background. Note that the ROI circle is also used as the initial contour in active contour detection algorithm. 
        - **use_active_contour** (*boolean, optional*): [Default=True] parameter that specifies whether to use active contour detection algorithm. If False, the contour used for image masking will simply be the ROI circle determined from ``roi_ratio``.
    Optional arguments inherited from :py:func:`skimage.segmentation.active_contour` (with different default values). See `scikit-image API <https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.active_contour>`_ for more information.:
        - **alpha** (*float, optional*): [Default=None] Hyper-parameter for active contour detection. Snake length shape parameter. Higher values makes snake contract faster. If None, the value of alpha will be automatically calculated from the size of ``recon_init``.
        - **beta** (*float, optional*): [Default=10.] Hyper-parameter for active contour detection. Snake smoothness shape parameter. Higher values makes snake smoother.
        - **w_line** (*float, optional*): [Default=-0.5] Hyper-parameter for active contour detection. Controls attraction to brightness. Use negative values to attract toward dark regions.
        - **w_edge** (*float, optional*): [Default=1.5] Hyper-parameter for active contour detection. Controls attraction to edges. Use negative values to repel snake from edges.
        - **gamma** (*float, optional*): [Default=0.01] Hyper-parameter for active contour detection. Explicit time stepping parameter.
    Optional arguments inherited from :py:func:`cone3D.recon` and :py:func:`cone3D.project`:
        - **det_channel_offset** (*float, optional*): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a row.
        - **det_row_offset** (*float, optional*): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a column.
        - **rotation_offset** (*float, optional*): [Default=0.0] Distance in :math:`ALU` from source-detector line to axis of rotation in the object space. This is normally set to zero.

        - **delta_pixel_detector** (*float, optional*): [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.
        - **delta_pixel_image** (*float*): [Default=None] Scalar value of image pixel spacing in :math:`ALU`. If None, automatically set to delta_pixel_detector/magnification
        - **ror_radius** (*float*): [Default=None] Scalar value of radius of reconstruction in :math:`ALU`. If None, automatically set with compute_img_params. Pixels outside the radius ror_radius in the :math:`(x,y)` plane are disregarded in the reconstruction.

        - **init_image** (*ndarray, optional*): [Default=0.0] Initial value of reconstruction image, specified by either a scalar value or a 3D numpy array with shape (num_img_slices,num_img_rows,num_img_cols)
        prox_image (ndarray, optional): [Default=None] 3D proximal map input image. 3D numpy array with shape (num_img_slices,num_img_rows,num_img_cols)

        - **sigma_y** (*float, optional*): [Default=None] Scalar value of noise standard deviation parameter. If None, automatically set with auto_sigma_y.
        - **snr_db** (*float, optional*): [Default=30.0] Scalar value that controls assumed signal-to-noise ratio of the data in dB. Ignored if sigma_y is not None.
        - **weights** (*ndarray, optional*): [Default=None] 3D weights array with same shape as sino.
        - **weight_type** (*string, optional*): [Default='unweighted'] Type of noise model used for data. If the ``weights`` array is not supplied, then the function ``cone3D.calc_weights`` is used to set weights using specified ``weight_type`` parameter.

                - Option "unweighted" corresponds to unweighted reconstruction;
                - Option "transmission" is the correct weighting for transmission CT with constant dosage;
                - Option "transmission_root" is commonly used with transmission CT data to improve image homogeneity;
                - Option "emission" is appropriate for emission CT data.

        - **positivity** (*bool, optional*): [Default=True] Boolean value that determines if positivity constraint is enforced. The positivity parameter defaults to True; however, it should be changed to False when used in applications that can generate negative image values.
        - **p** (*float, optional*): [Default=1.2] Scalar value in range :math:`[1,2]` that specifies the qGGMRF shape parameter.
        - **q** (*float, optional*): [Default=2.0] Scalar value in range :math:`[p,1]` that specifies the qGGMRF shape parameter.
        - **T** (*float, optional*): [Default=1.0] Scalar value :math:`>0` that specifies the qGGMRF threshold parameter.
        - **num_neighbors** (*int, optional*): [Default=6] Possible values are {26,18,6}. Number of neightbors in the qggmrf neighborhood. Higher number of neighbors result in a better regularization but a slower reconstruction.
        - **sharpness** (*float, optional*): [Default=0.0] Scalar value that controls level of sharpness in the reconstruction. ``sharpness=0.0`` is neutral; ``sharpness>0`` increases sharpness; ``sharpness<0`` reduces sharpness. Ignored if ``sigma_x`` is not None in qGGMRF mode, or if ``sigma_p`` is not None in proximal map mode.
        - **sigma_x** (*float, optional*): [Default=None] Scalar value :math:`>0` that specifies the qGGMRF scale parameter. Ignored if prox_image is not None. If None and prox_image is also None, automatically set with auto_sigma_x. Regularization should be controled with the ``sharpness`` parameter, but ``sigma_x`` can be set directly by expert users.
        - **sigma_p** (*float, optional*): [Default=None] Scalar value :math:`>0` that specifies the proximal map parameter. Ignored if prox_image is None. If None and proximal image is not None, automatically set with auto_sigma_p. Regularization should be controled with the ``sharpness`` parameter, but ``sigma_p`` can be set directly by expert users.
        - **max_iterations** (*int, optional*): [Default=20] Integer valued specifying the maximum number of iterations.
        - **stop_threshold** (*float, optional*): [Default=0.02] Scalar valued stopping threshold in percent. If stop_threshold=0.0, then run max iterations.
        - **num_threads** (*int, optional*): [Default=None] Number of compute threads requested when executed. If None, num_threads is set to the number of cores in the system
        - **NHICD** (*bool, optional*): [Default=False] If true, uses Non-homogeneous ICD updates
        - **verbose** (*int, optional*): [Default=1] Possible values are {0,1,2}, where 0 is quiet, 1 prints minimal reconstruction progress information, and 2 prints the full information.
        - **lib_path** (*str, optional*): [Default=~/.cache/mbircone] Path to directory containing library of forward projection matrices.
    Returns:
        2-element tuple containing:
        
        - **sino_corected** (*ndarray, float*): corrected 3D sinogram with shape (num_views, num_det_rows, num_det_channels)
        
        - **snake** (*ndarray, float*): 3D array containing the coordinates of contour generated from active contour detection algorithm.
    
   
    """
    # initial recon
    if recon_init is None:
        print("Performing inital qGGMRF reconstruction with uncorrected sinogram ......")
        recon_init = cone3D.recon(sino, angles, dist_source_detector, magnification, 
                         det_channel_offset=det_channel_offset, det_row_offset=det_row_offset, rotation_offset=rotation_offset,
                         delta_pixel_detector=delta_pixel_detector, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius,
                         init_image=init_image,
                         sigma_y=sigma_y, snr_db=snr_db, weights=weights, weight_type=weight_type,
                         positivity=positivity, p=p, q=q, T=T, num_neighbors=num_neighbors,
                         sharpness=sharpness, sigma_x=sigma_x, max_iterations=max_iterations, stop_threshold=stop_threshold,
                         num_threads=num_threads, NHICD=NHICD, verbose=verbose, lib_path=lib_path)
    # Image segmentation
    print("Performing image segmentation ......")
    mask = image_mask(recon_init, roi_ratio=roi_ratio, use_active_contour=use_active_contour, gauss_sigma=gauss_sigma, alpha=alpha, beta=beta, w_line=w_line, w_edge=w_edge, gamma=gamma)
    x_m = recon_init*mask
    (num_views, num_det_rows, num_det_channels) = np.shape(sino)
    print("Calculating sinogram error ......")
    Ax_m = cone3D.project(x_m, angles,
                          num_det_rows, num_det_channels,
                          dist_source_detector, magnification,
                          det_channel_offset=det_channel_offset, det_row_offset=det_row_offset, rotation_offset=rotation_offset,
                          delta_pixel_detector=delta_pixel_detector, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius,
                          num_threads=num_threads, verbose=verbose, lib_path=lib_path)
    # sinogram error
    e = sino-Ax_m
    p = np.array([gaussian(e[i], gauss_sigma, preserve_range=True) for i in range(num_views)])
    print("Linear fitting ......")
    c = np.sum(e*p) / np.sum(p*p)
    print("linear fitting constant = ", c)
    sino_corrected = sino - c*p
    return sino_corrected


def background_offset_calibration(sino, background_box_info_list):
    """Performs background offset calibration to the sinogram. 
    
    Args:
        sino (ndarray): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels)
        background_box_info_list ([(left,top,width,height,view_ind)]): [default=[]] A list of tuples specifying the rectangular areas used for background offset calculation.  
            Each tuple in the list has entries of the form `(left, top, width, height, view_ind)`.  Here `(left, top)` specifies the left top corner of the rectangle in pixels (using the convention that the left top of the entire image is (0,0)), `width` and `height` are also in pixels, and `view_ind` is the index of the view associated with this rectangle.    
    Returns:
        float: The background offset calculated from `sino`.
    """
    avg_offset = 0.
    if not background_box_info_list:
        return 0.
    for box_info in background_box_info_list:
        #print(f"box used for background offset calbiration: (x,y,width,height,view_ind)=", box_info)
        (x, y, box_width, box_height, view_ind) = box_info
        avg_offset += np.mean(sino[view_ind, y:y + box_height, x:x + box_width])
    avg_offset /= len(background_box_info_list)
    return avg_offset


def NSI_process_raw_scans(radiographs_directory, NSI_system_params,
                          blank_scan_path=None, dark_scan_path=None,
                          num_time_points=None, time_point=0,
                          num_sampled_scans=None,                      
                          rotation_direction="positive"):
    """Reads a subset of scan images corresponding to the given time point from an NSI ConeBeam scan directory, and calculates view angles corresponding to the object scans.
    
    Args:
        radiographs_directory (string): Path to an NSI ConeBeam scan directory.
        NSI_system_params (dict): A dictionary containing NSI parameters. This can be obtained from an NSI configuration file using function `preprocess.NSI_read_params()`.
        num_sampled_scans (int): [Default=None] Number of object scans sampled from all object scans. It should be smaller than the total number of object scans from directory. 
            By default, num_sampled_scans will be the total number of scans in the directory.
            The subset of the scans will be picked by a grid subsampling strategy. For example, by setting num_sampled_scans to be half of total number of object scans in the directory, the algorithm will pick every other object scan. 
        blank_scan_path (string): [Default=None] Path to a blank scan image, e.g. 'path_to_scan/gain0.tif'
        dark_scan_path (string): [Default=None] Path to a dark scan image, e.g. 'path_to_scan/offset.tif'
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
    angle_step = total_angles/num_acquired_scans
    if num_time_points is None:
        num_time_points = total_angles//360
        num_acquired_scans_cropped = int(num_time_points*360/angle_step)
        if num_acquired_scans != num_acquired_scans_cropped:
            print("Last rotation is incomplete! Discarding scans of the last rotation.")
            total_angles = num_acquired_scans_cropped*angle_step
    if num_sampled_scans is None:
        num_sampled_scans = num_acquired_scans_cropped
    scan_ids = _compute_views_index_list([0, num_acquired_scans_cropped], num_sampled_scans)
    view_ids = _select_contiguous_subset(scan_ids, num_time_points, time_point)
     
    angles = _compute_angles_list(view_ids, num_acquired_scans_cropped, total_angles, rotation_direction)
    obj_scan = _read_scan_dir(radiographs_directory, view_ids)

    # Should deal with situation when input is None.
    if blank_scan_path is not None:
        blank_scan = np.expand_dims(_read_scan_img(blank_scan_path), axis=0)
    else:
        blank_scan = np.expand_dims(0 * obj_scan[0] + 1, axis=0)
    if dark_scan_path is not None:
        dark_scan = np.expand_dims(_read_scan_img(dark_scan_path), axis=0)
    else:
        dark_scan = np.expand_dims(0 * obj_scan[0], axis=0)

    obj_scan = np.flip(obj_scan, axis=1)
    blank_scan = np.flip(blank_scan, axis=1)
    dark_scan = np.flip(dark_scan, axis=1)
    return obj_scan, blank_scan, dark_scan, angles


def compute_sino_from_scans(obj_scan, blank_scan=None, dark_scan=None,
                            downsample_factor=[1, 1], crop_factor=[(0, 0), (1, 1)],
                            weight_type='unweighted', defect_pixel_loc_list=None, defect_pixel_rot=2):
    """Given a set of object scan, blank scan, and dark scan, compute the sinogram used for reconstruction. This function will (optionally) downsample and crop the scans before computing the sinogram. It is assumed that the object scans, blank scan and dark scan all have compatible sizes. 
    
    Args:
        obj_scan (ndarray, float): 3D object scan with shape (num_views, num_det_rows, num_det_channels).
        blank_scan (ndarray, float): [Default=None] 3D blank scan with shape (num_blank_scans, num_det_rows, num_det_channels).
        dark_scan (ndarray, float): [Default=None] 3D dark scan with shape (num_dark_scans, num_det_rows, num_det_channels)
        downsample_factor ([int, int]): [Default=[1,1]] Two numbers to define down-sample factor.
        crop_factor ([(float, float),(float, float)] or [float, float, float, float]): [Default=[(0., 0.), (1., 1.)]].
            Two fractional points to define the bounding box. Sequence of [(r0, c0), (r1, c1)] or [r0, c0, r1, c1], where 0<=r0 <= r1<=1 and 0<=c0 <= c1<=1.
            In case where the scan size is not divisible by downsample_factor, the scans will be first truncated to a size that is divisible by downsample_factor, and then downsampled.
        weight_type (string, optional): [Default='unweighted'] Type of noise model used for data.
            The function ``cone3D.calc_weights`` is used to set weights using specified ``weight_type`` parameter.
                - Option "unweighted" corresponds to unweighted reconstruction;
                - Option "transmission" is the correct weighting for transmission CT with constant dosage;
                - Option "transmission_root" is commonly used with transmission CT data to improve image homogeneity;
                - Option "emission" is appropriate for emission CT data.
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
    # should add something here to check the validity of downsampled scan pixel values?
    sino, weight_mask = _compute_sino_and_weight_mask_from_scans(obj_scan, blank_scan, dark_scan)
    print('weight_mask shape = ', weight_mask.shape)
    # compute sinogram weights
    weights = cone3D.calc_weights(sino, weight_type=weight_type)
    # set the sino and weights corresponding to invalid sinogram entries to 0.
    weights[weight_mask == 0] = 0.
    sino[weight_mask == 0] = 0.
    # set defective pixel weights to be 0. 
    if defect_pixel_loc_list is not None:
        if crop_factor != [(0, 0), (1, 1)]:
            print("Defective pixel information ignored because radiograph is cropped.")
        else:
            print("Setting defective sinogram pixel weight to 0 ...")
            num_rot_forward = 4-defect_pixel_rot
            num_rot_backward = defect_pixel_rot
            weights = np.rot90(weights, num_rot_forward, axes=(1, 2))
            for (r,c) in defect_pixel_loc_list:
                r_ds = r//downsample_factor[0]
                c_ds = c//downsample_factor[1]
                weights[:,r_ds,c_ds]=False
            weights = np.rot90(weights, num_rot_backward, axes=(1, 2))
    return sino.astype(np.float32), weights.astype(np.float32)
