
import os
from glob import glob
import numpy as np
from PIL import Image
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import warnings


def nrmse(image, reference_image):
    """
    Compute the normalized root mean square error between image and reference_image.
    Args:
        image: Calculated image
        reference_image: Ground truth image
    Returns:
        Root mean square of (image - reference_image) divided by RMS of reference_image
    """
    rmse = np.sqrt(((image - reference_image) ** 2).mean())
    denominator = np.sqrt(((reference_image) ** 2).mean())

    return rmse/denominator


def read_ND(filePath, n_dim, dtype='float32', ntype='int32'):

    with open(filePath, 'rb') as fileID:

        sizesArray = np.fromfile( fileID, dtype=ntype, count=n_dim)
        numElements = np.prod(sizesArray)
        dataArray = np.fromfile(fileID, dtype=dtype, count=numElements).reshape(sizesArray)

    return dataArray


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


def downsample_scans(obj_scan, blank_scan, dark_scan, factor=1):

    assert len(factor)==2, 'factor({}) needs to be of len 2'.format(factor)

    new_size1 = factor[0]*(obj_scan.shape[1]//factor[0])
    new_size2 = factor[1]*(obj_scan.shape[2]//factor[1])

    obj_scan = obj_scan[:,0:new_size1,0:new_size2]
    blank_scan = blank_scan[:,0:new_size1,0:new_size2]
    dark_scan = dark_scan[:,0:new_size1,0:new_size2]

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
    index_sampled = [view_range[0]+np.int(np.floor(i*len(index_original)/num_views)) for i in range(num_views)]
    return index_sampled


def compute_angles_list( view_index_list, num_acquired_scans, total_angles,  rotation_direction="positive"):

    if rotation_direction not in ["positive",  "negative"]:
        warnings.warn("Parameter rotation_direction is not valid string; Setting center_offset = 'positive'.")
        rotation_direction = "positive"
    view_index_list = np.array(view_index_list)
    if rotation_direction == "positive":
        angles_list = ((2*np.pi/360) * total_angles * view_index_list / num_acquired_scans)% (2*np.pi)
    if rotation_direction == "negative":
        angles_list = (-(2*np.pi/360) * total_angles * view_index_list / num_acquired_scans)% (2*np.pi)

    return angles_list


def preprocess(path_radiographs, path_blank='gain0.tif', path_dark='offset.tif',
               view_range=[0,1999], angle_span=360, num_views=20, full_views=2000,
               rotation_direction="positive", downsample_factor=[4,4]):
    view_ids = compute_views_index_list(view_range, num_views)
    print(view_ids)
    angles = compute_angles_list(view_ids, full_views,angle_span, rotation_direction)
    obj_scan = read_scan_dir(path_radiographs, view_ids)
    if path_blank is not None:
        blank_scan = np.expand_dims(read_scan_img(path_blank), axis = 0)
    if path_dark is not None:
        dark_scan = np.expand_dims(read_scan_img(path_dark), axis = 0)

    # downsampling in views and pixels
    # obj_scan, blank_scan, dark_scan = downsample_scans(obj_scan, blank_scan, dark_scan,
    #                                     factor=downsample_factor)
    # obj_scan, blank_scan, dark_scan = crop_scans(obj_scan, blank_scan, dark_scan,
    #                                     limits_lo=dataset_params['crop_limits_lo'],
    #                                     limits_hi=dataset_params['crop_limits_hi'])

    sino = compute_sino(obj_scan, blank_scan, dark_scan)
    return sino.astype(np.float32), angles.astype(np.float32)


def test_1():
    """

    Test preprocessing without downsampling and cropping

    """
    dataset_dir = "/depot/bouman/users/li3120/datasets/metal_weld_data/"
    path_radiographs = dataset_dir+"Radiographs-2102414-2019-001-076/"
    sino, angles= preprocess(path_radiographs, path_blank=dataset_dir + 'Corrections/gain0.tif', path_dark=dataset_dir + 'Corrections/offset.tif',
                            view_range=[0, 1999], angle_span=360, num_views=13,rotation_direction="negative", downsample_factor=[1, 1])
    ref_sino = read_ND("./metal_laser_welds_cmp/object.sino", 3)
    ref_sino = np.copy(np.swapaxes(ref_sino, 1, 2))
    ref_sino = np.flip(ref_sino,axis = 1)
    np.save("sino.npy", sino)
    print(np.shape(sino))
    print(np.shape(ref_sino))
    plt.imshow(ref_sino[0]-sino[0])
    plt.colorbar()
    plt.savefig('diff_sino.png')
    plt.close()
    print(nrmse(sino[0], ref_sino[0]))
    print(angles)

    """
    Test result：
    (10, 1920, 1536)
    (10, 1920, 1536)
    0.0
    """

if __name__ == '__main__':
    test_1()

