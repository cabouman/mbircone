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
        blank_scan (ndarray) : A blank scan. 3D numpy array, (num_obj_scans, num_slices, num_channels).
        dark_scan (ndarray):  A dark scan. 3D numpy array, (num_obj_scans, num_slices, num_channels).
    Returns:
        A tuple (sino, weight_mask) containing:
        - **sino** (*ndarray*): Preprocessed sinogram with shape (num_views, num_slices, num_channels).
        - **weight_mask** (*ndarray*): A binary mask for sinogram weights. 

    """
    # take average of multiple blank/dark scans, and expand the dimension to be the same as obj_scan.
    blank_scan_mean = 0 * obj_scan + np.mean(blank_scan, axis=0, keepdims=True)
    dark_scan_mean = 0 * obj_scan + np.mean(dark_scan, axis=0, keepdims=True)

    obj_scan_corrected = (obj_scan - dark_scan_mean)
    blank_scan_corrected = (blank_scan_mean - dark_scan_mean)
    sino = -np.log(obj_scan_corrected / blank_scan_corrected)
    # weight_mask. 1 corresponds to valid sinogram values, and 0 corresponds to invalid sinogram values.
    # this will later be used to calculate sinogram weights in function compute_sino_from_scans.
    weight_mask = (obj_scan_corrected > 0) & (blank_scan_corrected > 0) 
    print('Set sinogram weight corresponding to nan and inf pixels to 0.')
    weight_mask[np.isnan(sino)] = False
    weight_mask[np.isinf(sino)] = False
    return sino, weight_mask

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
    """ Reads NSI specific geometry and sinogram parameters from an NSI configuration file.
    
    This function is specific to NSI datasets. 
    
    Args:
        config_file_path (string): Path to NSI configuration file. The filename extension is '.nsipro'.
    Returns:
        Dictionary containing NSI system parameters.
    """
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
    params = _NSI_read_str_from_config(config_file_path, tag_section_list)
    NSI_system_params = dict()

    # coordinate of source
    NSI_system_params['u_s'] = np.single(params[0].split(' ')[-1])

    # coordinate of reference
    vals = params[1].split(' ')
    NSI_system_params['u_d1'] = np.single(vals[2])
    NSI_system_params['v_d1'] = np.single(vals[0])
    NSI_system_params['w_d1'] = np.single(vals[1])
    
    # detector pixel pitch
    vals = params[2].split(' ')
    NSI_system_params['delta_dv'] = np.single(vals[0])
    NSI_system_params['delta_dw'] = np.single(vals[1])
    
    # dimension of radiograph
    NSI_system_params['N_dv'] = int(params[3])
    NSI_system_params['N_dw'] = int(params[4])
    
    # total number of radiograph scans
    NSI_system_params['num_acquired_scans'] = int(params[5])
    
    # total angles (usually 360 for 3D data, and (360*number_of_full_rotations) for 4D data
    NSI_system_params['total_angles'] = int(params[6])
    
    # Radiograph rotation (degree)
    NSI_system_params['rotate'] = int(params[7])
    if (NSI_system_params['rotate'] == 180) or (NSI_system_params['rotate'] == 0):
        print('scans are in portrait mode!')
    elif (NSI_system_params['rotate'] == 270) or (NSI_system_params['rotate'] == 90):
        print('scans are in landscape mode!')
        NSI_system_params['N_dv'], NSI_system_params['N_dw'] = NSI_system_params['N_dw'], NSI_system_params['N_dv']
    else:        
        warnings.warn("Picture mode unknown! Should be either portrait (0 or 180 deg rotation) or landscape (90 or 270 deg rotation). Automatically setting picture mode to portrait.")
        NSI_system_params['rotate'] = 180 
    
    # Radiograph horizontal & vertical flip
    if params[8] == "True":
        NSI_system_params['flipH'] = True
    else: 
        NSI_system_params['flipH'] = False
    if params[9] == "True":
        NSI_system_params['flipV'] = True
    else: 
        NSI_system_params['flipV'] = False
    
    # Detector rotation angle step (degree)
    NSI_system_params['angleStep'] = np.single(params[10])
    
    # Detector rotation direction
    if params[11] == "True":
        print("clockwise rotation.")
        NSI_system_params['rotation_direction'] = "positive"
    else:
        print("counter-clockwise rotation.")
        # counter-clockwise rotation
        NSI_system_params['rotation_direction'] = "negative"
    NSI_system_params['v_d0'] = - NSI_system_params['v_d1']
    NSI_system_params['w_d0'] = - NSI_system_params['N_dw'] * NSI_system_params['delta_dw'] / 2.0
    NSI_system_params['v_r'] = 0.0
    return NSI_system_params


def NSI_adjust_sysparam(NSI_system_params, downsample_factor=[1, 1], crop_factor=[(0, 0), (1, 1)]):
    """Returns adjusted NSI system parameters given downsampling factor and cropping factor.

    This function is specific to NSI datasets. 
    
    Args:
        NSI_system_params (dict of string-int): NSI system parameters.
        downsample_factor ([int, int]): [Default=[1,1]] Down-sample factors along the detector rows and channels respectively. See docstring of ``preprocess.NSI_process_raw_scans`` for more details.
        crop_factor ([(float, float),(float, float)] or [float, float, float, float]): [Default=[(0., 0.), (1., 1.)]].
            Two fractional points [(r0, c0), (r1, c1)] defining the bounding box that crops the scans. See docstring of ``preprocess.NSI_process_raw_scans`` for more details.
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

    This function is specific to NSI datasets. 
    
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


def NSI_process_raw_scans(obj_scan_path, blank_scan_path, dark_scan_path,
                          NSI_system_params,
                          downsample_factor=[1, 1], crop_factor=[(0, 0), (1, 1)],
                          view_id_start=0, view_angle_start=0., 
                          view_id_end=None, subsample_view_factor=1): 
    """ This function process the blank scan, dark scan, and object scan images according to the following procedures:
        
        1. Load the blank scan, dark scan, and a subset of object scan images according to ``view_id_start``, ``view_id_end``, and ``subsample_view_factor``.
        2. Rotate and flip the scans according to NSI parameters. 
        3. (optionally) crop and downsample the scan images according to ``crop_factor`` and ``downsample_factor``.
        4. Calculate the view angles corresponding to the object scan images.
        
    This function is specific to NSI datasets. 
    
    Args:
        obj_scan_path (string): Path to an NSI radiograph directory.
        blank_scan_path (string): [Default=None] Path to a blank scan image, e.g. 'path_to_scan/gain0.tif'
        dark_scan_path (string): [Default=None] Path to a dark scan image, e.g. 'path_to_scan/offset.tif'
        NSI_system_params (dict): A dictionary containing NSI parameters. This can be obtained from an NSI configuration file using function ``preprocess.NSI_read_params()``.

        downsample_factor ([int, int]): [Default=[1,1]] Down-sample factors along the detector rows and channels respectively.
            In case where the scan size is not divisible by `downsample_factor`, the scans will be first truncated to a size that is divisible by `downsample_factor`, and then downsampled.

        crop_factor ([(float, float),(float, float)] or [float, float, float, float]): [Default=[(0., 0.), (1., 1.)]].
            Two fractional points [(r0, c0), (r1, c1)] defining the bounding box that crops the scans, where 0<=r0<=r1<=1 and 0<=c0<=c1<=1.
            r0 and r1 defines the cropping factors along the detector rows. c0 and c1 defines the cropping factors along the detector channels. ::
            
            :       (0,0)--------------------------(0,1)
            :         |  (r0,c0)---------------+     |
            :         |     |                  |     |
            :         |     | (Cropped Region) |     |
            :         |     |                  |     |
            :         |     +---------------(r1,c1)  |
            :       (1,0)--------------------------(1,1)
            
            For example, ``crop_factor=[(0.25,0), (0.75,1)]`` will crop out the middle half of the scan image along the vertical direction. The cropped scan image will have half of the detector rows as before.
            By default no cropping or downsampling is performed.
        view_id_start (int): [Default=0] view id corresponding to the first object scan image.
        view_angle_start (float): [Default=0.0] view angle in radian corresponding to the first object scan image.
        view_id_end (int): [Default=None] view id corresponding to the last object scan image. If None, this will be equal to the total number of scan images in ``obj_scan_path``.
        subsample_view_factor (int): [Default=1]: view subsample factor. By default no view subsampling will be performed. 
            For example, with ``subsample_view_factor=2``, every other object scan will be loaded.

    Returns: 
        4-element tuple containing:
        
        - **obj_scan** (*ndarray, float*): 3D object scan with shape (num_views, num_det_rows, num_det_channels)
        
        - **blank_scan** (*ndarray, float*): 3D blank scan with shape (1, num_det_rows, num_det_channels)
        
        - **dark_scan** (*ndarray, float*): 3D dark scan with shape (1, num_det_rows, num_det_channels)
        
        - **angles** (*ndarray, double*): 1D view angles array in radians in the interval :math:`[0,2\pi)`.
    """
    # read blank scans and dark scans
    blank_scan = np.expand_dims(_read_scan_img(blank_scan_path), axis=0)
    dark_scan = np.expand_dims(_read_scan_img(dark_scan_path), axis=0)

    if view_id_end is None:
        view_id_end = NSI_system_params["num_acquired_scans"]
    view_ids = list(range(view_id_start, view_id_end, subsample_view_factor)) 
    obj_scan = _read_scan_dir(obj_scan_path, view_ids)
    
    # flip the scans according to flipH and flipV params 
    if NSI_system_params['flipV']:
        print("Flip scans vertically!")
        obj_scan = np.flip(obj_scan, axis=1)
        blank_scan = np.flip(blank_scan, axis=1)
        dark_scan = np.flip(dark_scan, axis=1)
    if NSI_system_params['flipH']:
        print("Flip scans horizontally!")
        obj_scan = np.flip(obj_scan, axis=2)
        blank_scan = np.flip(blank_scan, axis=2)
        dark_scan = np.flip(dark_scan, axis=2)

    # rotate the scans according to rotate param
    rot_count = NSI_system_params['rotate'] // 90
    obj_scan = np.rot90(obj_scan, rot_count, axes=(2,1))    
    blank_scan = np.rot90(blank_scan, rot_count, axes=(2,1))    
    dark_scan = np.rot90(dark_scan, rot_count, axes=(2,1))    
     
    # downsampling in pixels
    obj_scan, blank_scan, dark_scan = _downsample_scans(obj_scan, blank_scan, dark_scan,
                                                        downsample_factor=downsample_factor)
    # cropping in pixels
    obj_scan, blank_scan, dark_scan = _crop_scans(obj_scan, blank_scan, dark_scan,
                                                  crop_factor=crop_factor)
   
    # compute projection angles based on angleStep and rotation direction
    view_angle_start_deg = np.rad2deg(view_angle_start)
    angle_step = NSI_system_params['angleStep'] * subsample_view_factor
    if NSI_system_params['rotation_direction'] == "negative":
        angles = np.deg2rad(np.array([(view_angle_start_deg-n*angle_step) % 360.0 for n in range(len(view_ids))]))
    else:
        angles = np.deg2rad(np.array([(view_angle_start_deg+n*angle_step) % 360.0 for n in range(len(view_ids))]))
    return obj_scan, blank_scan, dark_scan, angles


def compute_sino_and_weight_from_scans(obj_scan, blank_scan, dark_scan,
                                       weight_type='unweighted'):
    """Given a set of object scans, blank scan, and dark scan, compute the sinogram data and weights. It is assumed that the object scans, blank scan and dark scan all have compatible sizes. 
    
    The sinogram values and weights corresponding to invalid sinogram entries will be set to 0.
 
    Args:
        obj_scan (ndarray, float): 3D object scan with shape (num_views, num_det_rows, num_det_channels).
        blank_scan (ndarray, float): [Default=None] 3D blank scan with shape (num_blank_scans, num_det_rows, num_det_channels).
        dark_scan (ndarray, float): [Default=None] 3D dark scan with shape (num_dark_scans, num_det_rows, num_det_channels)
        weight_type (string, optional): [Default='unweighted'] Type of noise model used for data. The function ``cone3D.calc_weights`` is used to set weights using specified ``weight_type`` parameter.
            - Option "unweighted" corresponds to unweighted reconstruction;
            - Option "transmission" is the correct weighting for transmission CT with constant dosage;
            - Option "transmission_root" is commonly used with transmission CT data to improve image homogeneity;
            - Option "emission" is appropriate for emission CT data.
    Returns:
        2-element tuple containing:
        - **sino** (*ndarray, float*): Preprocessed sinogram data with shape (num_views, num_det_rows, num_det_channels).
        - **weights** (*ndarray, float*): 3D weights array with the same shape as sino. 
    """

    # should add something here to check the validity of downsampled scan pixel values?
    sino, weight_mask = _compute_sino_and_weight_mask_from_scans(obj_scan, blank_scan, dark_scan)
    print('weight_mask shape = ', weight_mask.shape)
    # compute sinogram weights
    weights = cone3D.calc_weights(sino, weight_type=weight_type)
    # set the sino and weights corresponding to invalid sinogram entries to 0.
    weights[weight_mask == 0] = 0.
    sino[weight_mask == 0] = 0.
    return sino.astype(np.float32), weights.astype(np.float32)
