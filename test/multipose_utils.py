import numpy as np
import os
from mbircone.cone3D import project, backproject

__lib_path = os.path.join(os.path.expanduser('~'), '.cache', 'mbircone')

def distortion_matrix(recon, angles,
                      num_det_rows, num_det_channels,
                      dist_source_detector, magnification,
                      metal_threshold=0.1, background_threshold=0.01,
                      delta_det_channel=1.0, delta_det_row=1.0, delta_pixel_image=None,
                      det_channel_offset=0.0, det_row_offset=0.0, rotation_offset=0.0, image_slice_offset=0.0,
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
    D = backproject_metal / backproject_object
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
    D = np.clip(D, 0.0, 1.0)
    D = 1-D
    return D

