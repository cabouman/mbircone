import os, sys
import numpy as np
import math
import urllib.request
import tarfile
import mbircone
import demo_utils
import pprint
pp = pprint.PrettyPrinter(indent=4)

"""
This script is a demonstration of the metal artifact reduction (MAR) functionality in preprocess module. For more information please read the `[theory] <theory.html>`_ section in readthedocs.

Demo functionality includes:
 * downloading NSI dataset from specified urls;
 * Computing sinogram from object scan, blank scan, and dark scan images;
 * Perform an intial MBIR reconstruction;
 * Computing an adaptive data weight matrix from the sinogram and initial recon. This weight will be used to reduce metal artifacts.
 * Perform a MBIR recon with reduced metal artifacts;
 * Displaying the results.
"""
print('This script is a demonstration of the preprocessing module of NSI dataset. Demo functionality includes:\
\n\t * downloading NSI dataset from specified urls;\
\n\t * Loading object scans, blank scan, dark scan, view angles, and conebeam geometry parameters;\
\n\t * Computing sinogram and sino weights from object scan, blank scan, and dark scan images;\
\n\t * Computing a 3D reconstruction from the sinogram using a mbir prior model;\
\n\t * Displaying the results.\n')
print('This script is a demonstration of the metal artifact removal functionality in preprocess module. For more information please read the `[theory] <theory.html>`_ section in readthedocs.\
\nDemo functionality includes:\
\n\t * downloading NSI dataset from specified urls;\
\n\t * Computing sinogram from object scan, blank scan, and dark scan images;\
\n\t * Perform an intial MBIR reconstruction;\
\n\t * Computing an adaptive data weight matrix from the sinogram and initial recon. This weight will be used to reduce metal artifacts.\
\n\t * Perform a MBIR recon with reduced metal artifacts;\
\n\t * Displaying the results\n'.)

# ###########################################################################
# Set the parameters to get the data and do the recon 
# ###########################################################################

# ###################### Change the parameters below for your own use case.
# ##### params for dataset downloading.
# path to store output recon images
save_path = './output/mar_demo/'
os.makedirs(save_path, exist_ok=True)

# ##### Download and extract NSI dataset 
# url to NSI dataset.
dataset_url = 'https://engineering.purdue.edu/~bouman/data_repository/data/mar_demo_data.tgz'
# destination path to download and extract the phantom and NN weight files.
dataset_dir = './demo_data/'   
# download dataset. The dataset path will be later used to define path to NSI files.
dataset_path = demo_utils.download_and_extract(dataset_url, dataset_dir)

# ##### NSI specific file paths
# path to NSI config file. Change dataset path params for your own NSI dataset
nsi_config_file_path = os.path.join(dataset_path, 'mar_demo_data/JB-033_ArtifactPhantom_VerticalMetal.nsipro')
# path to directory containing all object scans
obj_scan_path = os.path.join(dataset_path, 'mar_demo_data/Radiographs-JB-033_ArtifactPhantom_VerticalMetal')
# path to blank scan. Usually <dataset_path>/Corrections/gain0.tif
blank_scan_path = os.path.join(dataset_path, 'mar_demo_data/Corrections/gain0.tif')
# path to dark scan. Usually <dataset_path>/Corrections/offset.tif
dark_scan_path = os.path.join(dataset_path, 'mar_demo_data/Corrections/offset.tif')
# downsample factor of scan images along detector rows and detector columns.
downsample_factor = [4, 4]
# ######### End of parameters #########

# ###########################################################################
# NSI preprocess: obtain sinogram, sino weights, angles, and geometry params
# ###########################################################################
print("\n*******************************************************",
      "\n*** Loading scan images, angles, and geometry params **",
      "\n*******************************************************")
obj_scan, blank_scan, dark_scan, angles, geo_params = \
        mbircone.preprocess.NSI_load_scans_and_params(nsi_config_file_path, obj_scan_path, 
                                                      blank_scan_path, dark_scan_path,
                                                      downsample_factor=downsample_factor)
print("MBIR geometry paramemters:")
pp.pprint(geo_params)
print('obj_scan shape = ', obj_scan.shape)
print('blank_scan shape = ', blank_scan.shape)
print('dark_scan shape = ', dark_scan.shape)

print("\n*******************************************************",
      "\n** Computing sino and sino weights from scan images ***",
      "\n*******************************************************")
sino, weights_unweighted = mbircone.preprocess.transmission_CT_preprocess(obj_scan, blank_scan, dark_scan)
print('sino shape = ', sino.shape)

# delete scan images to reduce memory usage
del obj_scan, blank_scan, dark_scan

# ###########################################################################
# Perform MBIR reconstruction
# ###########################################################################
print("\n***************************************************************",
      "\n****** Performing MBIR recon with unweighted sino weight ******",
      "\n********* This step will take 30-60 minutes to finish *********",
      "\n***************************************************************")

print("This recon will be used to identify metal voxels and compute the MAR sinogram weight.")
# extract mbircone geometry params required for recon
dist_source_detector = geo_params["dist_source_detector"]
magnification = geo_params["magnification"]
delta_det_row = geo_params["delta_det_row"]
delta_det_channel = geo_params["delta_det_channel"]
det_channel_offset = geo_params["det_channel_offset"]
det_row_offset = geo_params["det_row_offset"]

# MBIR recon
t_start = time.time()
recon_unweighted = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification,
                                         det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                         delta_det_row=delta_det_row, delta_det_channel=delta_det_channel,
                                         weights=weights_unweighted)
t_end = time.time()
t_elapsed = t_end - t_start
print(f"Recon with unweighted sino weight finished. Time elapsed {t_elapsed:.1f} sec.")


print("\n***************************************************************",
      "\n*************** Calculating MAR sinogram weight ***************",
      "\n***************************************************************")
weights_mar = mbircone.preprocess.calc_weight_mar(sino, init_recon=recon_unweighted, 
                                                  angles=angles, dist_source_detector=dist_source_detector, magnification=magnification,
                                                  metal_threshold=metal_threshold, 
                                                  good_pixel_mask=weights_unweighted
                                                  delta_det_channel=delta_det_channel, delta_det_row=delta_det_row,
                                                  det_channel_offset=det_channel_offset, det_row_offset=det_row_offset
                                                  )
# delete unweighted weight matrix to reduce memory usage
del weights_unweighted

print("\n***************************************************************",
      "\n******* Performing MBIR recon with MAR sinogram weight ********",
      "\n********* This step will take 30-60 minutes to finish *********",
      "\n***************************************************************")
t_start = time.time()
recon_mar = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification,
                                  det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                  delta_det_row=delta_det_row, delta_det_channel=delta_det_channel,
                                  weights=weights_mar, 
                                  # use previous recon as initialiation to save time
                                  init_image=recon_unweighted)
t_end = time.time()
t_elapsed = t_end - t_start
print(f"Recon with MAR weight finished. Time elapsed {t_elapsed:.1f} sec.")

print("\n*******************************************************",
      "\n**************** Plotting recon slices ****************",
      "\n*******************************************************")
view_angle_display = np.rad2deg(angles[0])
demo_utils.plot_image(recon_unweighted[:,:,149], title=f'MBIR recon with "unweighted" weight type',
                      filename=os.path.join(save_path, 'recon_unweighted_sagittal149.png'), vmin=0, vmax=0.055)

demo_utils.plot_image(recon_mar[:,:,149], title=f'MBIR recon with "MAR" weight type',
                      filename=os.path.join(save_path, 'recon_mar_sagittal149.png'), vmin=0, vmax=0.055)

input("press Enter")
