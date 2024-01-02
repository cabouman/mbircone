# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by MBIRCONE Developers
# All rights reserved. BSD 3-clause License.

import numpy as np
import os
import hashlib
import random
from PIL import Image
import h5py

def hdf5_save(image, filename="recon.h5", 
              source="", alu_def="", delta_pixel_image=1.0, recon_unit="ALU^{-1}"):
    """ Save a reconstruction image as an HDF5 file. The file structure is defined as follows:
        
        Dataset:
            - **voxels** (*ndarray*): 3D image to be saved.
        Attributes:
            - **README** (*string*): long string defining the structure of the HDF5 file.
            - **source** (*string*): source file of the reconstruction. Example: "reconstruction.nsihdr".
            - **alu_def** (*string*): definition of arbitrary length units (ALU). Example: "5 mm".
            - **delta_pixel_image** (*float*): Image pixel spacing in ALU.
            - **recon_unit** (*string*): unit of the reconstruction data. Example: "mm^{-1}"`.
    Args:
        image (float, ndarray): 3D image to be saved.
        filename (string, optional) [Default="recon.h5"] Path to save the HDF5 file.
        source (string, optional) [Default=""] source file of the reconstruction. Example: "reconstruction.nsihdr".
        alu_def (string, optional) [Default=""] definition of arbitrary length units (ALU). Example: "5 mm".
        delta_pixel_image (float, optional) [Default=1.0]: Image pixel spacing in ALU.
        recon_unit (string, optional): [Default="ALU^{-1}"] unit of the reconstruction data. Example: "mm^{-1}"
    """
    
    with h5py.File(filename, "w") as f:
        # voxel values
        f.create_dataset("voxels", data=image)
        # image shape
        f.attrs["source"] = source
        f.attrs["alu_def"] = alu_def
        f.attrs["delta_pixel_image"] = delta_pixel_image
        f.attrs["recon_unit"] = recon_unit
        f.attrs["README"] = \
            """
            The structure of this file is defined as follows:
                Dataset:
                - voxels (ndarray): 3D image to be saved.
                Attributes:
                - README (string): long string defining the structure of the HDF5 file.
                - source (string): source file of the reconstruction. Example: "reconstruction.nsihdr".
                - alu_def (string): definition of arbitrary length units (ALU). Example: "5 mm".
                - delta_pixel_image (float): Image pixel spacing in ALU.
                - recon_unit (string): unit of the reconstruction data. Example: "mm^{-1}"`.
            """
    return


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

