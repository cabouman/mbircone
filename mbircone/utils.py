# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by MBIRCONE Developers
# All rights reserved. BSD 3-clause License.

import numpy as np
import h5py

def hdf5_write(image, filename,
              recon_description="", alu_description ="", delta_pixel_image=1.0):
    """ This function writes a reconstructed image to an HDF5 file containing the 3D reconstructed volume along with optional descriptions of the source data, and units. 
            
    Args:
        image (float, ndarray): 3D reconstructed image to be saved.
        filename (string): Path to save the HDF5 file. Example: <path_to_directory>/recon.h5.
        recon_description (string, optional) [Default=""]: description of CT source data.
        alu_description (string, optional) [Default=""]: description of arbitrary length units (ALU). Example: "1 ALU = 5 mm".
        delta_pixel_image (float, optional) [Default=1.0]:  Image pixel spacing in ALU.
    """    
    f = h5py.File(filename, "w")
    # voxel values
    f.create_dataset("Data", data=image)
    # image shape
    f.attrs["recon_description"] = recon_description
    f.attrs["alu_description"] = alu_description
    f.attrs["delta_pixel_image"] = delta_pixel_image
    
    print("Attributes of HDF5 file: ")
    for k in f.attrs.keys():
        print(f"{k}: ", f.attrs[k])
    
    return
