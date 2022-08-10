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

def recon_resize_3D(recon, output_shape):
    """Resizes a reconstruction by performing 2D resizing along the slices dimension

    Args:
        recon (ndarray): 3D numpy array containing reconstruction with shape (slices, rows, cols)
        output_shape (tuple): (num_rows, num_cols) shape of resized output

    Returns:
        ndarray: 3D numpy array containing interpolated reconstruction with shape (num_slices, num_rows, num_cols).
    """

    Nz_in, Nx_in, Ny_in = recon.shape
    Nz_out, Nx_out, Ny_out = output_shape
    # define interp function from the input image
    zz_in, xx_in, yy_in = np.linspace(0, Nz_in-1, Nz_in), np.linspace(0, Nx_in-1, Nx_in), np.linspace(0, Ny_in-1, Ny_in)
    interp = RegularGridInterpolator((zz_in, xx_in, yy_in), recon)
    # define output image grid
    zz_out, xx_out, yy_out = np.linspace(0, Nz_in-1, Nz_out), np.linspace(0, Nx_in-1, Nx_out), np.linspace(0, Ny_in-1, Ny_out)
    Z_out, X_out, Y_out = np.meshgrid(zz_out, xx_out, yy_out, indexing='ij')
    # interpolation
    recon_interp = interp((Z_out, X_out, Y_out))
    return recon_interp
