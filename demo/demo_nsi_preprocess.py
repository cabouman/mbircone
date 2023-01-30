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
 * Computing a 3D reconstruction from the sinogram using a qGGMRF prior model;
 * Displaying the results.
"""
print('This script is a demonstration of the preprocessing module of NSI dataset. Demo functionality includes:\
\n\t * downloading NSI dataset from specified urls;\
\n\t * Loading object scans, blank scan, dark scan, view angles, and conebeam geometry parameters;\
\n\t * Computing sinogram and sino weights from object scan, blank scan, and dark scan images;\
\n\t * Computing a 3D reconstruction from the sinogram using a qGGMRF prior model;\
\n\t * Displaying the results.\n')

# ###########################################################################
# Set the parameters to get the data and do the recon 
# ###########################################################################

# Change the parameters below for your own use case.

# The url to the data repo.
data_repo_url = 'https://github.com/cabouman/mbir_data/raw/master/'

# Download url to the index file.
# This file will be used to retrieve urls to files that we are going to download
yaml_url = os.path.join(data_repo_url, 'index.yaml')

# Choice of NSI dataset. 
# These should be valid choices specified in the index file. 
# The url to the dataset will be parsed from data_repo_url and the choices of dataset specified below.
dataset_name = 'nsi_demo'

# destination path to download and extract the phantom and NN weight files.
dataset_dir = './demo_data/'   
# path to store output recon images
save_path = './output/nsi_demo/'
os.makedirs(save_path, exist_ok=True)

# ##### Download and extract data 
# Download the url index file and return path to local file. 
index_path = demo_utils.download_and_extract(yaml_url, dataset_dir) 
# Load the url index file as a directionary
url_index = demo_utils.load_yaml(index_path)
# get urls to phantom and denoiser parameter file
dataset_url = os.path.join(data_repo_url, url_index['dataset'][dataset_name])  # url to download the dataset
# download dataset. The dataset path will be later used to define path to NSI files.
dataset_path = demo_utils.download_and_extract(dataset_url, dataset_dir)

# ##### NSI specific file paths
# path to NSI config file. 
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
# Perform qGGMRF reconstruction
# ###########################################################################
print("\n*******************************************************",
      "\n*********** Performing MBIR reconstruction ************",
      "\n**** This step will take about one hour to finish *****",
      "\n*******************************************************")
# extract mbircone geometry params required for recon
dist_source_detector = geo_params["dist_source_detector"]
magnification = geo_params["magnification"]
delta_det_row = geo_params["delta_det_row"]
delta_det_channel = geo_params["delta_det_channel"]
det_channel_offset = geo_params["det_channel_offset"]
det_row_offset = geo_params["det_row_offset"]
# MBIR recon
recon_qGGMRF = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification,
                                     det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                     delta_det_row=delta_det_row, delta_det_channel=delta_det_channel,
                                     weights=weights)
print("qGGMRF recon finished. recon shape = ", np.shape(recon_qGGMRF))

print("\n*******************************************************",
      "\n********** Plotting sinogram images and gif ***********",
      "\n*******************************************************")
demo_utils.plot_gif(recon_qGGMRF, save_path, recon_file_name, vmin=vmin, vmax=vmax)
input("press Enter")
