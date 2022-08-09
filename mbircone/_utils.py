# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by SVMBIR Developers
# All rights reserved. BSD 3-clause License.

import numpy as np
import os
import hashlib
import random
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

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

def recon_resize_3D_old(recon, output_shape):
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
    if input_shape[0] < output_shape[0]:
        recon_resized = interp_3D(recon, output_shape)
    else:
        recon_resized = block_avg_3D(recon)    
    assert recon_resized.shape == output_shape, 'Error in _utils.recon_resize_3D: Resized image shape is incorrect!!'
    return recon_resized


def block_avg_3D(recon):
    num_slices, num_rows, num_cols = recon.shape
    num_slices_pad, num_rows_pad, num_cols_pad = int(np.ceil(num_slices/2)*2), int(np.ceil(num_rows/2)*2), int(np.ceil(num_cols/2)*2)
    recon_pad = np.empty((num_slices_pad, num_rows_pad, num_cols_pad))
    recon_pad[:num_slices, :num_rows, :num_cols] = recon
    
    # reflective boundary condition
    if num_slices%2 != 0:
        recon_pad[num_slices_pad-1,:,:] = recon[num_slices-1,:,:]
    if num_rows%2 != 0:
        recon_pad[:, num_rows_pad-1,:] = recon[:, num_rows-1,:]
    if num_cols%2 != 0:
        recon_pad[:, :, num_cols_pad-1] = recon[:, :, num_cols-1]

    print('recon_pad shape before block_avg = ', recon_pad.shape)
    recon_ds = recon_pad.reshape(recon_pad.shape[0]//2, 2, recon_pad.shape[1]//2, 2, recon_pad.shape[2]//2, 2).mean((1, 3, 5))
    print('recon_pad shape after block_avg = ', recon_ds.shape)
    return recon_ds

def interp_3D(recon, interp_shape):
    Nz_ds, Nx_ds, Ny_ds = recon.shape
    Nz, Nx, Ny = interp_shape
    X, Y, Z = np.linspace(0, Nz-1, num=Nz_ds, endpoint=True), np.linspace(0, Nx-1, num=Nx_ds, endpoint=True), np.linspace(0, Ny-1, num=Ny_ds, endpoint=True)
    interpolating_function = RegularGridInterpolator((X, Y, Z), recon)
    recon_interp = interpolating_function([(iz, ix, iy) for iz, ix, iy in np.ndindex(Nz, Nx, Ny)])
    recon_interp = recon_interp.reshape((Nz, Nx, Ny))
    return recon_interp
