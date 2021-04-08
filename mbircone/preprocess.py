
import os
import argparse
from glob import glob
import numpy as np
from PIL import Image
from skimage.measure import block_reduce
import scipy

from utils import *

# from mbir import *

# from multiprocessing import Pool
# import matplotlib.pyplot as plt


def read_scan_img(img_path):

    img = np.asarray(Image.open(img_path))

    if np.issubdtype(img.dtype, np.integer):
        # make float and normalize integer types
        maxval = np.iinfo(img.dtype).max
        img = img.astype(np.float32)/np.iinfo(img.dtype).max

    return img

def read_scan_dir(scan_dir, view_ids=None):

    img_path_list = sorted(glob(os.path.join(scan_dir, '*')))

    if view_ids!=None:
        img_path_list = [img_path_list[i] for i in view_ids]

    # print(img_path_list[0])
    # print(img_path_list[-1])
    
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

def downsample_scans(obj_scan, blank_scan, dark_scan, factor=1):

    assert len(factor)==2, 'factor({}) needs to be of len 2'.format(factor)

    new_size1 = factor[0]*(obj_scan.shape[0]//factor[0])
    new_size2 = factor[1]*(obj_scan.shape[1]//factor[1])

    obj_scan = obj_scan[0:new_size1,0:new_size2]
    blank_scan = blank_scan[0:new_size1,0:new_size2]
    dark_scan = dark_scan[0:new_size1,0:new_size2]

    obj_scan = block_reduce(obj_scan, block_size=(1, factor[0], factor[1]), func=np.sum)
    blank_scan = block_reduce(blank_scan, block_size=(1, factor[0], factor[1]), func=np.sum)
    dark_scan = block_reduce(dark_scan, block_size=(1, factor[0], factor[1]), func=np.sum)

    return obj_scan, blank_scan, dark_scan

def crop_scans(obj_scan, blank_scan, dark_scan, limits_lo=[0,0], limits_hi=[1,1]):

    assert len(limits_lo)==2, 'limits_lo needs to be of len 2'
    assert len(limits_hi)==2, 'limits_hi needs to be of len 2'
    assert limits_lo[1]==1-limits_hi[1], 'horizontal crop limits must be symmetric'

    N1_lo = round(limits_lo[0]*obj_scan.shape[1])
    N2_lo = round(limits_lo[1]*obj_scan.shape[2])

    N1_hi = round(limits_hi[0]*obj_scan.shape[1])
    N2_hi = round(limits_hi[1]*obj_scan.shape[2])

    obj_scan = obj_scan[:,N1_lo:N1_hi,N2_lo:N2_hi]
    blank_scan = blank_scan[:,N1_lo:N1_hi,N2_lo:N2_hi]
    dark_scan = dark_scan[:,N1_lo:N1_hi,N2_lo:N2_hi]
    
    return obj_scan, blank_scan, dark_scan

def compute_sino_wght(obj_scan, blank_scan, dark_scan, isClip=True):

    blank_scan_mean = 0*obj_scan + np.average(blank_scan, axis=0 )
    dark_scan_mean = 0*obj_scan + np.average(dark_scan, axis=0 )

    obj_scan_corrected = (obj_scan-dark_scan_mean)
    blank_scan_corrected = (blank_scan_mean-dark_scan_mean)

    good_pixels = (obj_scan_corrected>0) & (blank_scan_corrected>0)

    normalized_scan = np.zeros(obj_scan_corrected.shape)
    normalized_scan[good_pixels] = obj_scan_corrected[good_pixels]/blank_scan_corrected[good_pixels]

    sino = np.zeros(obj_scan_corrected.shape)
    sino[normalized_scan>0] = -np.log(normalized_scan[normalized_scan>0])

    wght = normalized_scan

    return sino, wght

def gen_view_ids(view_range, num_views):

    index_original = range(view_range[0], view_range[1]+1)

    assert num_views<=len(index_original), 'num_views cannot exceed range of view index'

    index_sampled = [ view_range[0]+round(i*len(index_original)/num_views) for i in range(num_views) ]

    return index_sampled

def gen_angles_full(angle_span, num_views):

    return np.pi*(angle_span/180.0)*np.array(range(0,num_views))/num_views


def preprocess(dataset_params):

    view_ids = gen_view_ids(dataset_params['view_range'], dataset_params['num_views'])

    angles = gen_angles_full(dataset_params['angle_span'], dataset_params['num_views'])

    obj_scan, blank_scan, dark_scan = read_scans_all(dataset_params['obj_scan_dir'], 
                                        blank_scan_dir=dataset_params['blank_scan_dir'], 
                                        dark_scan_dir=dataset_params['dark_scan_dir'], 
                                        view_ids=view_ids)

    if dataset_params['zinger']['median_filter_size']:
        obj_scan_median = scipy.signal.medfilt(obj_scan, kernel_size=dataset_params['zinger']['median_filter_size'])
        blank_scan_median = scipy.signal.medfilt(blank_scan, kernel_size=dataset_params['zinger']['median_filter_size'])
        dark_scan_median = scipy.signal.medfilt(dark_scan, kernel_size=dataset_params['zinger']['median_filter_size'])

        obj_scan[obj_scan>dataset_params['zinger']['threshold']] = obj_scan_median[obj_scan>dataset_params['zinger']['threshold']] 
        blank_scan[blank_scan>dataset_params['zinger']['threshold']] = blank_scan_median[blank_scan>dataset_params['zinger']['threshold']] 
        dark_scan[dark_scan>dataset_params['zinger']['threshold']] = dark_scan_median[dark_scan>dataset_params['zinger']['threshold']] 

    
    obj_scan, blank_scan, dark_scan = downsample_scans(obj_scan, blank_scan, dark_scan, 
                                        factor=dataset_params['downsample_factor'])

    obj_scan, blank_scan, dark_scan = crop_scans(obj_scan, blank_scan, dark_scan, 
                                        limits_lo=dataset_params['crop_limits_lo'], 
                                        limits_hi=dataset_params['crop_limits_hi'])

    sino, wght = compute_sino_wght(obj_scan, blank_scan, dark_scan)


    data_dict = {}
    data_dict['obj_scan'] = obj_scan
    data_dict['blank_scan'] = blank_scan
    data_dict['dark_scan'] = dark_scan
    data_dict['sino'] = sino
    data_dict['wght'] = wght
    data_dict['angles'] = angles

    return data_dict

def write_mbir_params(dataset_params, data_dict, rootPath):

    imgparams_fname = rootPath+'.imgparams'
    sinoparams_fname = rootPath+'.sinoparams'
    ViewAngleList_fname = rootPath+'.ViewAngleList'

    imgparams = {}
    imgparams['Nx'] = data_dict['sino'].shape[2]
    imgparams['Ny'] = data_dict['sino'].shape[2]
    imgparams['Nz'] = data_dict['sino'].shape[1]
    imgparams['FirstSliceNumber'] = 0
    imgparams['Deltaxy'] = dataset_params['pixel_sizes'][1]*dataset_params['downsample_factor'][1]
    imgparams['DeltaZ'] = dataset_params['pixel_sizes'][0]*dataset_params['downsample_factor'][0]
    imgparams['ROIRadius'] = imgparams['Nx'] * imgparams['Deltaxy'] / 2

    sinoparams = {}
    sinoparams['NChannels'] = data_dict['sino'].shape[2]
    sinoparams['NViews'] = len(data_dict['angles'])
    sinoparams['NSlices'] = data_dict['sino'].shape[1]
    sinoparams['DeltaChannel'] = dataset_params['pixel_sizes'][1]*dataset_params['downsample_factor'][1]
    sinoparams['CenterOffset'] = dataset_params['CenterOffset']/dataset_params['downsample_factor'][1]
    sinoparams['DeltaSlice'] = dataset_params['pixel_sizes'][0]*dataset_params['downsample_factor'][0]
    sinoparams['FirstSliceNumber'] = 0
    sinoparams['ViewAngleList'] = dataset_params['object_name']+'.ViewAngleList'

    modify_params(imgparams_fname, **imgparams)
    modify_params(sinoparams_fname, **sinoparams)

    with open(ViewAngleList_fname,'w') as fileID:
        for angle in list(data_dict['angles']):
            fileID.write(str(angle)+"\n")


def preprocess_conebeam(path_radiographs, path_blank='gain0.tif', path_dark='offset.tif', 
    view_range=[0,1999], angle_span=360, num_views=500, downsample_factor=[4,4]):

    obj_scan = 

    blank_scan = 

    dark_scan = 

    # downsampling in views and pixels

    def downsample_scans(obj_scan, blank_scan, dark_scan, factor=1)

    sino, wght = compute_sino_wght(obj_scan, blank_scan, dark_scan)
    
    return sino, wght


def main(args):

    dataset_params = read_params(args['dataset_params_path'])
    print_params(dataset_params)

    data_dict = preprocess(dataset_params)

    write_sino3D(data_dict['obj_scan'], os.path.join(args['scan_write_path'],'obj_scan.sino'))
    write_sino3D(data_dict['blank_scan'], os.path.join(args['scan_write_path'],'blank_scan.sino'))
    write_sino3D(data_dict['dark_scan'], os.path.join(args['scan_write_path'],'dark_scan.sino'))
    write_sino3D(data_dict['sino'], os.path.join(args['scan_write_path'],'sino.sino'))
    write_sino3D(data_dict['wght'], os.path.join(args['scan_write_path'],'wght.sino'))

    write_sino_openmbir(data_dict['sino'], os.path.join(args['mbir_data_path'],'sino',dataset_params['object_name']+'_slice'), '.2Dsinodata')
    write_sino_openmbir(data_dict['wght'], os.path.join(args['mbir_data_path'],'weight',dataset_params['object_name']+'_slice'), '.2Dweightdata')

    write_mbir_params(dataset_params, data_dict, args['mbir_params_path']+dataset_params['object_name'])


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dataset_params_path', dest='dataset_params_path', default='../params/dataset_params.yml')
arg_parser.add_argument('--mbir_params_path', dest='mbir_params_path', default='../params/sv-mbirct/')
arg_parser.add_argument('--mbir_data_path', dest='mbir_data_path', default='../data/sv-mbirct/')
arg_parser.add_argument('--scan_write_path', dest='scan_write_path', default='../data/scan/')
args, extra = arg_parser.parse_known_args()
args = vars(args)

if __name__ == '__main__':
    main(args)
