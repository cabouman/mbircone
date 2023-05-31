import os, sys
import numpy as np
import math
import urllib.request
import tarfile
import mbircone
import test_utils
import pprint
pp = pprint.PrettyPrinter(indent=4)

"""
This test script demonstrates the effect of defective pixels to MBIR reconstructions. This test generates two sets of reconstructions. The first reconstruction uses sino weights = 1.0 for all sinogram entries, while the second reconstruction sets the sino weights corresponding to defective pixels to 0.0.

The user should observe that the first reconstruction contains artifacts due to defective pixels. The artifacts are removed in the second recconstruction. 

This test takes about two hours to finish.
"""

print("This test script demonstrates the effect of defective pixels to MBIR reconstructions.\
\nThis test generates two sets of reconstructions with the full resolution sinogram data:\
\n\t * The first reconstruction uses sino weights = 1.0 for all sinogram entries.\
\n\t * The second reconstruction sets the sino weights corresponding to defective pixels to 0.0.\
\n The user should observe that the first reconstruction contains artifacts due to defective pixels, while the artifacts are removed in the second recconstruction.\
\n This test takes about two hours to finish." 
)
# ###########################################################################
# Set the parameters to get the data and do the recon 
# ###########################################################################

# ###################### Change the parameters below for your own use case.
# ##### params for dataset downloading.
# path to store output recon images
save_path = './output/test_defective_pixels/'
os.makedirs(save_path, exist_ok=True)

# ##### Download and extract NSI dataset 
# url to NSI dataset.
dataset_url = 'https://engineering.purdue.edu/~bouman/data_repository/data/demo_data_nsi.tgz'
# destination path to download and extract the phantom and NN weight files.
dataset_dir = './demo_data/'   
# download dataset. The dataset path will be later used to define path to NSI files.
dataset_path = test_utils.download_and_extract(dataset_url, dataset_dir)

# ##### NSI specific file paths
# path to NSI config file. Change dataset path params for your own NSI dataset
nsi_config_file_path = os.path.join(dataset_path, 'demo_data_nsi/JB-033_ArtifactPhantom_Vertical_NoMetal.nsipro')
# path to directory containing all object scans
obj_scan_path = os.path.join(dataset_path, 'demo_data_nsi/Radiographs-JB-033_ArtifactPhantom_Vertical_NoMetal')
# path to blank scan. Usually <dataset_path>/Corrections/gain0.tif
blank_scan_path = os.path.join(dataset_path, 'demo_data_nsi/Corrections/gain0.tif')
# path to dark scan. Usually <dataset_path>/Corrections/offset.tif
dark_scan_path = os.path.join(dataset_path, 'demo_data_nsi/Corrections/offset.tif')
# path to NSI file containing defective pixel information
defective_pixel_path = os.path.join(dataset_path, 'demo_data_nsi/Corrections/defective_pixels.defect')
# downsample factor of scan images along detector rows and detector columns.
downsample_sino_factor = [1, 1]
# downsample factor of image voxel pitch.
downsample_image_factor = 4
# ######### End of parameters #########

# ###########################################################################
# NSI preprocess: obtain sinogram, sino weights, angles, and geometry params
# ###########################################################################
print("\n*******************************************************",
      "\n*** Loading scan images, angles, and geometry params **",
      "\n*******************************************************")
obj_scan, blank_scan, dark_scan, angles, geo_params, defective_pixel_list = \
        mbircone.preprocess.NSI_load_scans_and_params(nsi_config_file_path, obj_scan_path, 
                                                      blank_scan_path, dark_scan_path,
                                                      downsample_factor=downsample_sino_factor,
                                                      defective_pixel_path=defective_pixel_path
                                                     )
print("MBIR geometry paramemters:")
pp.pprint(geo_params)
print('obj_scan shape = ', obj_scan.shape)
print('blank_scan shape = ', blank_scan.shape)
print('dark_scan shape = ', dark_scan.shape)

print("\n*******************************************************",
      "\n******** Computing sinogram from scan images **********",
      "\n*******************************************************")
sino, defective_pixel_list = \
        mbircone.preprocess.transmission_CT_compute_sino(obj_scan, blank_scan, dark_scan,
                                                         defective_pixel_list
                                                        )
# delete scan images to optimize memory usage
del obj_scan, blank_scan, dark_scan

print("\n*******************************************************",
      "\n************ Background offset correction *************",
      "\n*******************************************************")
background_offset = mbircone.preprocess.calc_background_offset(sino)
print("background_offset = ", background_offset)
sino = sino - background_offset

print("\n*******************************************************",
      "\n************** Calculate sinogram weight **************",
      "\n*******************************************************")

print("sino weight of MBIR recon #1: set all sinogram entries to 1.0")
weights_1 = mbircone.preprocess.calc_weights(sino, weight_type="unweighted")

print("sino weight of MBIR recon #2: set sinogram entries w.r.t. defective pixels to 0.0")
weights_2 = mbircone.preprocess.calc_weights(sino, weight_type="unweighted",
                                                   defective_pixel_list=defective_pixel_list
                                            )
# ###########################################################################
# Perform MBIR reconstruction
# ###########################################################################
# extract mbircone geometry params required for recon
dist_source_detector = geo_params["dist_source_detector"]
magnification = geo_params["magnification"]
delta_det_row = geo_params["delta_det_row"]
delta_det_channel = geo_params["delta_det_channel"]
det_channel_offset = geo_params["det_channel_offset"]
det_row_offset = geo_params["det_row_offset"]
# image_voxel_pitch = downsample_image_factor * default_voxel_pitch
delta_pixel_image = downsample_image_factor * delta_det_row / magnification

print("\n*************************************************************************",
      "\n**** MBIR reconstruction #1: sino weights = 1.0 for all sino entries ****",
      "\n******************* This step will take 40-80 minutes to finish *********",
      "\n*************************************************************************")

# MBIR recon
recon_mbir_1 = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification,
                                     det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                     delta_det_row=delta_det_row, delta_det_channel=delta_det_channel,
                                     delta_pixel_image=delta_pixel_image,
                                     weights=weights_1)
np.save(os.path.join(save_path, "recon_mbir_case_1.npy"), recon_mbir_1)
print("MBIR recon #1 finished. recon shape = ", np.shape(recon_mbir_1))

print("\n***************************************************************************",
      "\n** MBIR reconstruction #2: sino weights = 0.0 for defective sino entries **",
      "\n******************* This step will take 40-80 minutes to finish ***********",
      "\n***************************************************************************")

recon_mbir_2 = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification,
                                     det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                     delta_det_row=delta_det_row, delta_det_channel=delta_det_channel,
                                     delta_pixel_image=delta_pixel_image,
                                     weights=weights_2)
np.save(os.path.join(save_path, "recon_mbir_case_2.npy"), recon_mbir_2)
print("MBIR recon #2 finished. recon shape = ", np.shape(recon_mbir_2))

print("\n*******************************************************",
      "\n******** Plotting sinogram view and recon slices ******",
      "\n*******************************************************")
test_utils.plot_image(recon_mbir_1[419,:,:], title=f'MBIR recon case #1 (all sino weight = 1.0), axial slice 419',
                      filename=os.path.join(save_path, 'recon_1_axial419.png'), vmin=0, vmax=0.055)
test_utils.plot_image(recon_mbir_2[419,:,:], title=f'MBIR recon case #2 (defective sino weight = 0.0), axial slice 419',
                      filename=os.path.join(save_path, 'recon_2_axial419.png'), vmin=0, vmax=0.055)

input("press Enter")
