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


def _downsample_scans(obj_scan, blank_scan, dark_scan, downsample_factor=[1, 1]):
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

    return obj_scan, blank_scan, dark_scan


def _compute_sino_and_weight_mask_from_scans(obj_scan, blank_scan, dark_scan):
    """Computes sinogram data and weights mask base on given object scan, blank scan, and dark scan. The weights mask is used to filter out negative values in the corrected object scan and blank scan. For real CT dataset weights mask should be used when calculating sinogram weights.
    
    Args:
        obj_scan (ndarray): A stack of sinograms. 3D numpy array, (num_views, num_det_rows, num_det_channels).
        blank_scan (ndarray) : A blank scan. 3D numpy array, (num_obj_scans, num_det_rows, num_det_channels).
        dark_scan (ndarray):  A dark scan. 3D numpy array, (num_obj_scans, num_det_rows, num_det_channels).
    Returns:
        A tuple (sino, weight_mask) containing:
        - **sino** (*ndarray*): Preprocessed sinogram with shape (num_views, num_det_rows, num_det_channels).
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
                              downsample_factor=[1, 1], crop_factor=[(0, 0), (1, 1)],
                              view_id_start=0, view_angle_start=0., 
                              view_id_end=None, subsample_view_factor=1):
    """ Load the object scan, blank scan, dark scan, view angles, and geometry parameters from an NSI dataset directory.
     
    The scan images will be (optionally) cropped and downsampled.

    A subset of the views may be selected based on user input. In that case, the object scan images and view angles corresponding to the subset of the views will be returned.

    This function is specific to NSI datasets. 
    
    **Arguments specific to file paths**:
        
        - config_file_path (string): Path to NSI configuration file. The filename extension is '.nsipro'.
        - obj_scan_path (string): Path to an NSI radiograph directory.
        - blank_scan_path (string): [Default=None] Path to a blank scan image, e.g. 'path_to_scan/gain0.tif'
        - dark_scan_path (string): [Default=None] Path to a dark scan image, e.g. 'path_to_scan/offset.tif'
    
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
        5-element tuple containing:

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
    
    # flip the scans according to flipH and flipV NSI_params 
    if flipV:
        print("Flip scans vertically!")
        obj_scan = np.flip(obj_scan, axis=1)
        blank_scan = np.flip(blank_scan, axis=1)
        dark_scan = np.flip(dark_scan, axis=1)
    if flipH:
        print("Flip scans horizontally!")
        obj_scan = np.flip(obj_scan, axis=2)
        blank_scan = np.flip(blank_scan, axis=2)
        dark_scan = np.flip(dark_scan, axis=2)

    # rotate the scans according to scan_rotate param
    rot_count = scan_rotate // 90
    obj_scan = np.rot90(obj_scan, rot_count, axes=(2,1))    
    blank_scan = np.rot90(blank_scan, rot_count, axes=(2,1))    
    dark_scan = np.rot90(dark_scan, rot_count, axes=(2,1))    
     
    # downsampling in pixels
    obj_scan, blank_scan, dark_scan = _downsample_scans(obj_scan, blank_scan, dark_scan,
                                                        downsample_factor=downsample_factor)
    # cropping in pixels
    obj_scan, blank_scan, dark_scan = _crop_scans(obj_scan, blank_scan, dark_scan,
                                                  crop_factor=crop_factor)
   
    # compute projection angles based on angle_step and rotation direction
    view_angle_start_deg = np.rad2deg(view_angle_start)
    angle_step *= subsample_view_factor
    angles = np.deg2rad(np.array([(view_angle_start_deg+n*angle_step) % 360.0 for n in range(len(view_ids))]))
    return obj_scan, blank_scan, dark_scan, angles, geo_params


def transmission_CT_preprocess(obj_scan, blank_scan, dark_scan,
                               weight_type='unweighted'):
    """Given a set of object scans, blank scan, and dark scan, compute the sinogram data and weights. It is assumed that the object scans, blank scan and dark scan all have compatible sizes. 
    
    The weights and sinogram values corresponding to invalid sinogram entries will be set to 0.0.
 
    Args:
        obj_scan (ndarray, float): 3D object scan with shape (num_views, num_det_rows, num_det_channels). 
        blank_scan (ndarray, float): [Default=None] 3D blank scan with shape (num_blank_scans, num_det_rows, num_det_channels). When num_blank_scans>1, the pixel-wise mean will be used as the blank scan.
        dark_scan (ndarray, float): [Default=None] 3D dark scan with shape (num_dark_scans, num_det_rows, num_det_channels). When num_dark_scans>1, the pixel-wise mean will be used as the dark scan.
        weight_type (string, optional): [Default='unweighted'] Type of noise model used for data.

                - ``'unweighted'`` corresponds to unweighted reconstruction;
                - ``'transmission'`` is the correct weighting for transmission CT with constant dosage;
                - ``'transmission_root'`` is commonly used with transmission CT data to improve image homogeneity;

    Returns:
        2-element tuple containing:
        - **sino** (*ndarray, float*): Preprocessed sinogram data with shape (num_views, num_det_rows, num_det_channels).
        - **weights** (*ndarray, float*): 3D weights array with the same shape as sino. ``weights`` = 0.0 indicates an invalid sinogram entry in ``sino``.
    """

    # should add something here to check the validity of downsampled scan pixel values?
    sino, weight_mask = _compute_sino_and_weight_mask_from_scans(obj_scan, blank_scan, dark_scan)
    # set the sino corresponding to invalid entries to 0.0
    sino[weight_mask == 0] = 0.0
    # compute sinogram weights
    weights = cone3D.calc_weights(sino, weight_type=weight_type)
    # set the sino weights corresponding to invalid entries to 0.0
    weights[weight_mask == 0] = 0.0
    return sino.astype(np.float32), weights.astype(np.float32)


def calc_weight_mar(sino, init_recon, 
                    angles, dist_source_detector, magnification,
                    metal_threshold, good_pixel_mask, 
                    beta=2.0, gamma=4.0,
                    delta_det_channel=1.0, delta_det_row=1.0, delta_pixel_image=None,
                    det_channel_offset=0.0, det_row_offset=0.0, rotation_offset=0.0, image_slice_offset=0.0,
                    num_threads=None, verbose=0, lib_path=__lib_path):
    """ Compute the weights used for reducing metal artifacts in MBIR reconstruction. For more information please refer to the `[theory] <theory.html>`_ section in readthedocs.
    
    Required arguments:
        - **sino** (*ndarray*): Sinogram data with 3D shape (num_det_rows, num_det_channels).
        - **init_recon** (*ndarray*): Initial reconstruction used to identify metal voxels.
        - **angles** (*ndarray*): 1D array of view angles in radians.
        - **dist_source_detector** (*float*): Distance between the X-ray source and the detector in units of :math:`ALU`.
        - **magnification** (*float*): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
        - **metal_threshold** (*float*): Threshold value in units of :math:`ALU^{-1}` used to identify metal voxels. Any voxels in ``init_recon`` with an attenuation coefficient larger than ``metal_threshold`` will be identified as a metal voxel.
        - **good_pixel_mask** (*ndarray*): pixel mask specifying the location of valid sinogram entries.
            
            0.0 indicates an invalid pixel in the associated entry of ``sino``.
    
    Arguments specific to MAR data weights: 
        - **beta** (*float, optional*): [Default=2.0] Scalar value in range :math:`>0`.
            
            ``beta`` controls the weight to sinogram entries with low photon counts.
            A larger ``beta`` value improves image homogeneity, but may result in more severe metal artifacts.
        
        - **gamma** (*float, optional*): [Default=4.0] Scalar value in range :math:`>1`.
            
            ``gamma`` controls the weight to sinogram entries in which the projection paths contain metal components.
            A larger ``gamma`` value reduces image artifacts around metal regions, but may result in worse image quality inside metal regions, as well as reduced image homogeneity.
    
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
   
    # metal mask
    metal_mask = recon_init > metal_threshold
    _, num_det_rows, num_det_channels = sino.shape
    # sino_mask: 1 if projection path contains metal voxels, 0 else.
    sino_mask = cone3D.project(metal_mask.astype(float), angles,
                               num_det_rows=num_det_rows, num_det_channels=num_det_channels,
                               dist_source_detector=dist_source_detector, magnification=magnification,
                               delta_det_channel=delta_det_channel, delta_det_row=delta_det_row, delta_pixel_image=delta_pixel_image,
                               det_channel_offset=det_channel_offset, det_row_offset=det_row_offset, rotation_offset=rotation_offset, image_slice_offset=image_slice_offset,
                               num_threads=num_threads, verbose=verbose, lib_path=__lib_path)
    
    weights = np.zeros(sino.shape)
    # case where projection path does not contain metal voxels: weight = np.exp(-sino/beta)
    weights[sino_mask<=0] = np.exp(-sino[sino_mask<=0] / beta)
    # case where projection path contains metal voxels: weight = np.exp(-sino*gamma/beta)
    weights[sino_mask>0] = np.exp(-sino[sino_mask>0] * gamma / beta)
    weights[good_pixel_mask == 0.0] = 0.0
    return weights
