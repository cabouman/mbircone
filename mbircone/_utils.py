# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SVMBIR Developers
# All rights reserved. BSD 3-clause License.

import numpy as np
import os
import hashlib
import random
from PIL import Image


def hash_params(angles, sinoparams, imgparams):
    hash_input = str(sinoparams) + str(imgparams) + str(np.around(angles, decimals=6))
    hash_val = hashlib.sha512(hash_input.encode()).hexdigest()
    return hash_val

def _gen_sysmatrix_fname(lib_path, sysmatrix_name='object'):
    os.makedirs(os.path.join(lib_path, 'sysmatrix'), exist_ok=True)

    sysmatrix_fname = os.path.join(lib_path, 'sysmatrix', sysmatrix_name + '.sysmatrix')

    return sysmatrix_fname


def _gen_sysmatrix_fname_tmp(lib_path, sysmatrix_name='object'):
    sysmatrix_fname_tmp = os.path.join(lib_path, 'sysmatrix',
                                       sysmatrix_name + '_pid' + str(os.getpid()) + '_rndnum' + str(
                                           random.randint(0, 1000)) + '.sysmatrix')

    return sysmatrix_fname_tmp

def recon_resize_2D(recon, output_shape):
    """Resizes a reconstruction by performing 2D resizing along the slices dimension

    Args:
        recon (ndarray): 3D numpy array containing reconstruction with shape (slices, rows, cols)
        output_shape (tuple): (num_rows, num_cols) shape of resized output

    Returns:
        ndarray: 3D numpy array containing interpolated reconstruction with shape (num_slices, num_rows, num_cols).
    """

    recon_resized = np.empty((recon.shape[0],output_shape[0],output_shape[1]), dtype=recon.dtype)
    for i in range(recon.shape[0]):
        PIL_image = Image.fromarray(recon[i])
        PIL_image_resized = PIL_image.resize((output_shape[1],output_shape[0]), resample=Image.Resampling.BILINEAR)
        recon_resized[i] = np.array(PIL_image_resized)

    return recon_resized


def recon_resize_3D(recon, output_shape):
    """Resizes a reconstruction by performing 3D resizing along the horizontal and vertical dimensions.

    Args:
        recon (ndarray): 3D numpy array containing reconstruction with shape (num_slices_in, num_rows_in, num_cols_in)
        output_shape (tuple): (num_slices_out, num_rows_out, num_cols_out) shape of resized output

    Returns:
        ndarray: 3D numpy array containing interpolated reconstruction with shape (num_slices_out, num_rows_out, num_cols_out).
    """
    # 2D resize in horizontal plane (num_slices unchanged)
    input_shape = recon.shape
    recon_resized_horizontal = np.empty((input_shape[0],output_shape[1],output_shape[2]), dtype=recon.dtype)
    for i in range(input_shape[0]):
        PIL_image = Image.fromarray(recon[i])
        PIL_image_resized = PIL_image.resize((output_shape[2],output_shape[1]), resample=Image.Resampling.BILINEAR)
        recon_resized_horizontal[i] = np.array(PIL_image_resized)
    
    # 2D resize in vertical plane
    recon_resized = np.empty((output_shape[0],output_shape[1],output_shape[2]), dtype=recon.dtype)
    for j in range(output_shape[1]):
        PIL_image = Image.fromarray(recon_resized_horizontal[:, j, :])
        PIL_image_resized = PIL_image.resize((output_shape[2],output_shape[0]), resample=Image.Resampling.BILINEAR) # shape: num_slices x num_cols
        recon_resized[:,j,:] = np.array(PIL_image_resized)
    return recon_resized

