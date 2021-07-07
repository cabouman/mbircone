
import os
from glob import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import math


def read_scan_img(img_path):

    img = np.asarray(Image.open(img_path))

    if np.issubdtype(img.dtype, np.integer):
        # make float and normalize integer types
        maxval = np.iinfo(img.dtype).max
        img = img.astype(np.float32)/maxval

    return img.astype(np.float32)


def read_scan_dir(scan_dir, view_ids=None):

    img_path_list = sorted(glob(os.path.join(scan_dir, '*')))

    if view_ids!=None:
        img_path_list = [img_path_list[i] for i in view_ids]

    img_list = [ read_scan_img(img_path) for img_path in img_path_list]

    # return shape = N_theta x N_z x N_y
    return np.stack(img_list, axis=0)


def read_scans_all(obj_scan_dir, blank_scan_dir='', dark_scan_dir='', view_ids=None):

    obj_scan = read_scan_dir(obj_scan_dir, view_ids)

    if not blank_scan_dir:
        blank_scan = 0*obj_scan[0]+1
        blank_scan = blank_scan.reshape([1,obj_scan.shape[1],obj_scan.shape[2]])
    else:        
        blank_scan = read_scan_dir(blank_scan_dir)

    if not dark_scan_dir:
        dark_scan = 0*obj_scan[0]
        blank_scan = blank_scan.reshape([1,obj_scan.shape[1],obj_scan.shape[2]])
    else:        
        dark_scan = read_scan_dir(dark_scan_dir)

    return obj_scan, blank_scan, dark_scan



def downsample_scans(obj_scan, blank_scan, dark_scan, factor=[1,1]):

    assert len(factor)==2, 'factor({}) needs to be of len 2'.format(factor)

    new_size1 = factor[0]*(obj_scan.shape[1]//factor[0])
    new_size2 = factor[1]*(obj_scan.shape[2]//factor[1])

    obj_scan   = obj_scan[:,0:new_size1,0:new_size2]
    blank_scan = blank_scan[:,0:new_size1,0:new_size2]
    dark_scan  = dark_scan[:,0:new_size1,0:new_size2]

    obj_scan   = obj_scan.reshape(obj_scan.shape[0], obj_scan.shape[1]//factor[0], factor[0], obj_scan.shape[2]//factor[1], factor[1]).sum((2,4))
    blank_scan = blank_scan.reshape(blank_scan.shape[0], blank_scan.shape[1]//factor[0], factor[0], blank_scan.shape[2]//factor[1], factor[1]).sum((2,4))
    dark_scan  = dark_scan.reshape(dark_scan.shape[0], dark_scan.shape[1]//factor[0], factor[0], dark_scan.shape[2]//factor[1], factor[1]).sum((2,4))

    return obj_scan, blank_scan, dark_scan


def crop_scans(obj_scan, blank_scan, dark_scan, limits_lo=[0,0], limits_hi=[1,1]):
    assert len(limits_lo)==2, 'limits_lo needs to be of len 2'
    assert len(limits_hi)==2, 'limits_hi needs to be of len 2'
    assert math.isclose(limits_lo[1],1-limits_hi[1]), 'horizontal crop limits must be symmetric'

    N1_lo = round(limits_lo[0]*obj_scan.shape[1])
    N2_lo = round(limits_lo[1]*obj_scan.shape[2])

    N1_hi = round(limits_hi[0]*obj_scan.shape[1])
    N2_hi = round(limits_hi[1]*obj_scan.shape[2])

    obj_scan = obj_scan[:,N1_lo:N1_hi,N2_lo:N2_hi]
    blank_scan = blank_scan[:,N1_lo:N1_hi,N2_lo:N2_hi]
    dark_scan = dark_scan[:,N1_lo:N1_hi,N2_lo:N2_hi]
    
    return obj_scan, blank_scan, dark_scan


def compute_sino(obj_scan, blank_scan, dark_scan):

    blank_scan_mean = 0*obj_scan + np.average(blank_scan, axis=0 )
    dark_scan_mean = 0*obj_scan + np.average(dark_scan, axis=0 )

    obj_scan_corrected = (obj_scan-dark_scan_mean)
    blank_scan_corrected = (blank_scan_mean-dark_scan_mean)

    good_pixels = (obj_scan_corrected>0) & (blank_scan_corrected>0)

    normalized_scan = np.zeros(obj_scan_corrected.shape)
    normalized_scan[good_pixels] = obj_scan_corrected[good_pixels]/blank_scan_corrected[good_pixels]

    sino = np.zeros(obj_scan_corrected.shape)
    sino[normalized_scan>0] = -np.log(normalized_scan[normalized_scan>0])

    return sino


def compute_views_index_list(view_range, num_views):

    index_original = range(view_range[0], view_range[1]+1)
    assert num_views <= len(index_original), 'num_views cannot exceed range of view index'
    index_sampled = [view_range[0]+int(np.floor(i*len(index_original)/num_views)) for i in range(num_views)]
    return index_sampled


def select_contiguousSubset(indexList, num_chunk=1, ind_chunk=0):
    assert ind_chunk<num_chunk, 'ind_chunk cannot be larger than num_chunk.'

    len_ind =len(indexList)
    num_per_set = len_ind//num_chunk

    # distribute the remaining
    remaining_index_num = len_ind % num_chunk 

    start_id = ind_chunk*num_per_set+np.minimum(ind_chunk,remaining_index_num)
    end_id = (ind_chunk+1)*num_per_set+np.minimum(ind_chunk+1,remaining_index_num)

    indexList_new = indexList[start_id:end_id]
    return indexList_new


def compute_angles_list( view_index_list, num_acquired_scans, total_angles,  rotation_direction="positive"):

    if rotation_direction not in ["positive",  "negative"]:
        warnings.warn("Parameter rotation_direction is not valid string; Setting center_offset = 'positive'.")
        rotation_direction = "positive"
    view_index_list = np.array(view_index_list)
    if rotation_direction == "positive":
        angles_list = ( total_angles * view_index_list / num_acquired_scans)%360.0/360.0* (2*np.pi)
    if rotation_direction == "negative":
        angles_list = ( -total_angles * view_index_list / num_acquired_scans)%360.0/360.0* (2*np.pi)

    return angles_list


def read_NSI_string(filepath, tags_sections):

    tag_strs = ['<'+tag+'>' for tag,section in tags_sections]
    section_starts = ['<'+section+'>' for tag,section in tags_sections]
    section_ends = ['</'+section+'>'for tag,section in tags_sections]
    params = []

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except IOError:
         print ("Could not read file:", filepath)

    for tag_str, section_start, section_end in zip(tag_strs, section_starts, section_ends):
        section_start_inds = [ind for ind,match in enumerate(lines) if section_start in match]
        section_end_inds = [ind for ind,match in enumerate(lines) if section_end in match]
        section_start_ind = section_start_inds[0]
        section_end_ind = section_end_inds[0]

        for line_ind in range(section_start_ind+1,section_end_ind):
            line = lines[line_ind]
            if tag_str in line:
                tag_ind = line.find(tag_str,1)+len(tag_str)
                if tag_ind == -1:
                    params.append("")
                else:
                    params.append(line[tag_ind:].strip('\n'))

    return params

def read_NSI_params(filepath):
    tag_section_list = [['source','Result'],
                        ['reference','Result'],
                        ['pitch','Object Radiograph'],
                        ['width pixels','Detector'],
                        ['height pixels','Detector'],
                        ['number','Object Radiograph'],
                        ['Rotation range','CT Project Configuration']]
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
    NSI_system_params['w_d0'] =  - NSI_system_params['N_dw'] * NSI_system_params['delta_dw']  / 2.0
    NSI_system_params['v_r'] = 0.0
    return NSI_system_params


def adjust_NSI_sysparam(NSI_system_params, downsample_factor=[1,1], crop_factor=[[0,0],[1,1]]):
    # Adjust parameters after downsampling
    NSI_system_params['N_dw'] = (NSI_system_params['N_dw']//downsample_factor[0])
    NSI_system_params['N_dv'] = (NSI_system_params['N_dv']//downsample_factor[1])

    NSI_system_params['delta_dw'] =  NSI_system_params['delta_dw'] * downsample_factor[0];
    NSI_system_params['delta_dv'] =  NSI_system_params['delta_dv'] * downsample_factor[1];


    # Adjust parameters after cropping

    N_dwshift0 = np.round(NSI_system_params['N_dw'] * crop_factor[0][0]);
    N_dwshift1 = np.round(NSI_system_params['N_dw']* (1-crop_factor[1][0]));
    NSI_system_params['w_d0'] = NSI_system_params['w_d0']+ N_dwshift0 * NSI_system_params['delta_dw'];
    NSI_system_params['N_dw'] = NSI_system_params['N_dw'] - (N_dwshift0 + N_dwshift1);

    N_dvshift0 = np.round(NSI_system_params['N_dv'] * crop_factor[0][1]);
    N_dvshift1 = np.round(NSI_system_params['N_dv']* (1-crop_factor[1][1]));
    NSI_system_params['v_d0'] = NSI_system_params['v_d0']+ N_dvshift0 * NSI_system_params['delta_dv'];
    NSI_system_params['N_dv'] = NSI_system_params['N_dv'] - (N_dvshift0 + N_dvshift1);

    return NSI_system_params


def transfer_NSI_to_MBIRCONE(NSI_system_params):

    geo_params=dict()
    geo_params["num_channels"] = NSI_system_params['N_dv']
    geo_params["num_slices"] = NSI_system_params['N_dw']
    geo_params["delta_pixel_detector"] = NSI_system_params['delta_dv']
    geo_params["rotation_offset"] = NSI_system_params['v_r']

    geo_params["dist_source_detector"] = NSI_system_params['u_d1'] - NSI_system_params['u_s']
    geo_params["magnification"] = -geo_params["dist_source_detector"]/NSI_system_params['u_s']

    dist_dv_to_detector_corner_from_detector_center = - NSI_system_params['N_dv']*NSI_system_params['delta_dw']/2.0
    dist_dw_to_detector_corner_from_detector_center = - NSI_system_params['N_dw']*NSI_system_params['delta_dv']/2.0
    geo_params["channel_offset"] = -(NSI_system_params['v_d0']-dist_dv_to_detector_corner_from_detector_center)
    geo_params["row_offset"] = - (NSI_system_params['w_d0']-dist_dw_to_detector_corner_from_detector_center)
    return geo_params


def preprocess(path_radiographs, path_blank, path_dark,
               view_range=[0,360], num_views=20, total_angles=360,  num_acquired_scans=2000,
               rotation_direction="positive", downsample_factor=[1,1], crop_factor=[[0,0],[1,1]],
               num_time_points=1,index_time_points=0):
    view_ids = compute_views_index_list(view_range, num_views)
    view_ids = select_contiguousSubset(view_ids, num_time_points, index_time_points)
    angles = compute_angles_list(view_ids, num_acquired_scans, total_angles, rotation_direction)
    obj_scan = read_scan_dir(path_radiographs, view_ids)

    # Should deal with situation when input is None.
    if path_blank is not None:
        blank_scan = np.expand_dims(read_scan_img(path_blank), axis = 0)
    if path_dark is not None:
        dark_scan = np.expand_dims(read_scan_img(path_dark), axis = 0)

    obj_scan = np.flip(obj_scan, axis=1)
    blank_scan = np.flip(blank_scan, axis=1)
    dark_scan = np.flip(dark_scan, axis=1)

    # downsampling in views and pixels
    obj_scan, blank_scan, dark_scan = downsample_scans(obj_scan, blank_scan, dark_scan,
                                        factor=downsample_factor)
    obj_scan, blank_scan, dark_scan = crop_scans(obj_scan, blank_scan, dark_scan,
                                        limits_lo=crop_factor[0],
                                        limits_hi=crop_factor[1])

    sino = compute_sino(obj_scan, blank_scan, dark_scan)
    return sino.astype(np.float32), angles.astype(np.float64)

