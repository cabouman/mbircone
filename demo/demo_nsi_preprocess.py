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
This script is a demonstration of the preprocessing module of NSI dataset. Demo functionality includes:
 * downloading NSI dataset from specified urls;
 * Loading object scans, blank scan, dark scan, view angles, and conebeam geometry parameters;
 * Computing sinogram and sino weights from object scan, blank scan, and dark scan images;
 * Computing a 3D reconstruction from the sinogram using a mbir prior model;
 * Displaying the results.
"""
print('This script is a demonstration of the preprocessing module of NSI dataset. Demo functionality includes:\
\n\t * downloading NSI dataset from specified urls;\
\n\t * Loading object scans, blank scan, dark scan, view angles, and conebeam geometry parameters;\
\n\t * Computing sinogram and sino weights from object scan, blank scan, and dark scan images;\
\n\t * Computing a 3D reconstruction from the sinogram using a mbir prior model;\
\n\t * Displaying the results.\n')

# ###########################################################################
# Set the parameters to get the data and do the recon 
# ###########################################################################

# ###################### Change the parameters below for your own use case.
# ##### params for dataset downloading.
# path to store output recon images
save_path = './output/nsi_demo/'
os.makedirs(save_path, exist_ok=True)

# ##### Download and extract NSI dataset 
# url to NSI dataset.
dataset_url = 'https://engineering.purdue.edu/~bouman/data_repository/data/demo_data_nsi.tgz'
# destination path to download and extract the phantom and NN weight files.
dataset_dir = './demo_data/'   
# download dataset. The dataset path will be later used to define path to NSI files.
dataset_path = demo_utils.download_and_extract(dataset_url, dataset_dir)

# ##### NSI specific file paths
# path to NSI config file. Change dataset path params for your own NSI dataset
nsi_config_file_path = os.path.join(dataset_path, 'demo_data_nsi/JB-033_ArtifactPhantom_Vertical_NoMetal.nsipro')
# path to directory containing all object scans
obj_scan_path = os.path.join(dataset_path, 'demo_data_nsi/Radiographs-JB-033_ArtifactPhantom_Vertical_NoMetal')
# path to blank scan. Usually <dataset_path>/Corrections/gain0.tif
blank_scan_path = os.path.join(dataset_path, 'demo_data_nsi/Corrections/gain0.tif')
# path to dark scan. Usually <dataset_path>/Corrections/offset.tif
dark_scan_path = os.path.join(dataset_path, 'demo_data_nsi/Corrections/offset.tif')
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
sino, weights = mbircone.preprocess.transmission_CT_preprocess(obj_scan, blank_scan, dark_scan)
print('sino shape = ', sino.shape)

# ###########################################################################
# Perform MBIR reconstruction
# ###########################################################################
print("\n*******************************************************",
      "\n*********** Performing MBIR reconstruction ************",
      "\n**** This step will take 30-60 minutes to finish ******",
      "\n*******************************************************")
# extract mbircone geometry params required for recon
dist_source_detector = geo_params["dist_source_detector"]
magnification = geo_params["magnification"]
delta_det_row = geo_params["delta_det_row"]
delta_det_channel = geo_params["delta_det_channel"]
det_channel_offset = geo_params["det_channel_offset"]
det_row_offset = geo_params["det_row_offset"]
# MBIR recon
recon_mbir = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification,
                                   det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                   delta_det_row=delta_det_row, delta_det_channel=delta_det_channel,
                                   weights=weights)
print("MBIR recon finished. recon shape = ", np.shape(recon_mbir))

print("\n*******************************************************",
      "\n******** Plotting sinogram view and recon slices ******",
      "\n*******************************************************")
view_angle_display = np.rad2deg(angles[0])
demo_utils.plot_image(sino[0], title=f'Sinogram, view angle {view_angle_display:.1f} deg',
                      filename=os.path.join(save_path, 'sino_view0.png'), vmin=0, vmax=1.0)
demo_utils.plot_image(recon_mbir[210,:,:], title=f'MBIR recon, axial slice 210',
                      filename=os.path.join(save_path, 'recon_axial210.png'), vmin=0, vmax=0.055)
demo_utils.plot_image(recon_mbir[:,187,:], title=f'MBIR recon, coronal slice 187',
                      filename=os.path.join(save_path, 'recon_coronal187.png'), vmin=0, vmax=0.055)
demo_utils.plot_image(recon_mbir[:,:,198], title=f'MBIR recon, sagittal slice 198',
                      filename=os.path.join(save_path, 'recon_sagittal198.png'), vmin=0, vmax=0.055)

input("press Enter")
