import numpy as np
import os
import SimpleITK as sitk
from mbircone.cone3D import project, backproject
from transform_utils import transformer_sitk
from scipy.special import softmax

__lib_path = os.path.join(os.path.expanduser('~'), '.cache', 'mbircone')

def distortion_matrix(recon, angles,
                      num_det_rows, num_det_channels,
                      dist_source_detector, magnification,
                      metal_threshold=0.1, background_threshold=0.01,
                      delta_det_channel=1.0, delta_det_row=1.0, delta_pixel_image=None,
                      det_channel_offset=0.0, det_row_offset=0.0, rotation_offset=0.0, image_slice_offset=0.0,
                      epsilon=1e-6,
                      num_threads=None, verbose=1, lib_path=__lib_path):

    """ Given a reconstruction, compute its distortion matrix as D_i = 1 - [A^tAb_m]_i/[A^tAb_o]_i
    """
    num_image_slices, num_image_rows, num_image_cols = recon.shape
    
    # step 1: compute metal mask and object mask
    print("######### step 1: compute metal mask and object mask #########")
    b_metal = (recon > metal_threshold).astype(float)
    b_object = (recon > background_threshold).astype(float)
    
    # step 2: compute Ab_m and Ab_o
    print("######### step 2: compute Ab_m and Ab_o #########")
    Ab_m = project(b_metal, angles,
                   num_det_rows, num_det_channels,
                   dist_source_detector, magnification,
                   delta_det_channel=delta_det_channel, delta_det_row=delta_det_row, delta_pixel_image=delta_pixel_image,
                   det_channel_offset=det_channel_offset, det_row_offset=det_row_offset, rotation_offset=rotation_offset, image_slice_offset=image_slice_offset,
                   num_threads=num_threads, verbose=verbose, lib_path=lib_path)

    Ab_o = project(b_object, angles,
                   num_det_rows, num_det_channels,
                   dist_source_detector, magnification,
                   delta_det_channel=delta_det_channel, delta_det_row=delta_det_row, delta_pixel_image=delta_pixel_image,
                   det_channel_offset=det_channel_offset, det_row_offset=det_row_offset, rotation_offset=rotation_offset, image_slice_offset=image_slice_offset,
                   num_threads=num_threads, verbose=verbose, lib_path=lib_path)

    # step 3: compute AtAb_m and AtAb_o
    print("######### step 3: compute AtAb_m and AtAb_o #########")
    backproject_metal = backproject(Ab_m, angles,
                                    dist_source_detector, magnification,
                                    num_image_rows=num_image_rows, num_image_cols=num_image_cols, num_image_slices=num_image_slices,
                                    delta_det_channel=delta_det_channel, delta_det_row=delta_det_row, delta_pixel_image=delta_pixel_image,
                                    det_channel_offset=det_channel_offset, det_row_offset=det_row_offset, rotation_offset=rotation_offset, image_slice_offset=image_slice_offset,
                                    num_threads=num_threads, verbose=verbose, lib_path=lib_path)
     
    backproject_object = backproject(Ab_o, angles,
                                     dist_source_detector, magnification,
                                     num_image_rows=num_image_rows, num_image_cols=num_image_cols, num_image_slices=num_image_slices,
                                     delta_det_channel=delta_det_channel, delta_det_row=delta_det_row, delta_pixel_image=delta_pixel_image,
                                     det_channel_offset=det_channel_offset, det_row_offset=det_row_offset, rotation_offset=rotation_offset, image_slice_offset=image_slice_offset,
                                     num_threads=num_threads, verbose=verbose, lib_path=lib_path)
            
    # step 4: compute AtAb_m / AtAb_o, and replace nan/inf with 0
    print("######### step 4: compute AtAb_m / AtAb_o #########")
    D = backproject_metal / (backproject_object + epsilon)
    D = np.clip(D, 0.0, 1.0)
    return D


def weighted_average(recon_list, distortion_matrix_list, transformation_list, ref_image, alpha=50):
    num_poses = len(recon_list)
    assert(len(distortion_matrix_list) == num_poses and len(transformation_list) == num_poses) , "Error! length of recon_list, distortion_matrix_list, transformation_list must be the same!"
    recon_transformed_list = []
    D_transformed_list = [] 
    for recon, D, transformation in zip(recon_list, distortion_matrix_list, transformation_list):
        recon_transformed_list.append(transformer_sitk(recon, transformation, ref_image=ref_image))
        D_transformed_list.append(transformer_sitk(D, transformation, ref_image=ref_image, cval=1.0))
    

    M = softmax(-alpha*np.array(D_transformed_list), axis=0)
    recon_weighted_avg = np.sum(np.array([M[i]*recon_transformed_list[i] for i in range(num_poses)]), axis=0)
    return recon_weighted_avg, M
