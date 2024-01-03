# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by MBIRCONE Developers
# All rights reserved. BSD 3-clause License.

import numpy as np
import h5py

def hdf5_write(image, filename, 
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
        filename (string) Path to save the HDF5 file. Example: "<path_to_directory>/recon.h5"
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


def hdf5_read(filename):
    """ Read the image data as well as its metadata from an HDF5 file. The HDF5 file is assumed to have the following structure:
        Dataset:
            - **voxels** (*ndarray*): 3D image data.
        Attributes:
            - **README** (*string*): long string defining the structure of the HDF5 file.
            - **source** (*string*): source file of the reconstruction. Example: "reconstruction.nsihdr".
            - **alu_def** (*string*): definition of arbitrary length units (ALU). Example: "5 mm".
            - **delta_pixel_image** (*float*): Image pixel spacing in ALU.
            - **recon_unit** (*string*): unit of the reconstruction data. Example: "mm^{-1}"`.

    Args:
        filename (string, optional) [Default="recon.h5"] Path to save the HDF5 file.
    
    Returns:
        two-element tuple containing:

        - **image** (*ndarray*): 3D image data.
        - **metadata** (*dict*): A dictionary containing metadata of the image.
    """
    
    f = h5py.File(filename, "r")
    image = f["voxels"]  
    metadata = {}
    for k in f.attrs.keys():
        metadata[k] = f.attrs[k]
    return image, metadata
