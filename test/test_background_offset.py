import os, sys
import numpy as np
import math
import urllib.request
import tarfile
import mbircone
import test_utils
import pprint
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)

"""
This script tests the background offset calculation functionality with an NSI dataset. The test includes:
 * downloading NSI dataset from specified urls;
 * Loading object scans, blank scan, dark scan, view angles, and conebeam geometry parameters;
 * Computing sinogram from object scan, blank scan, and dark scan images;
 * Performing background offset correction to the sinogram data;
 * Displaying a sinogram view before and after the background offset correction.
"""

print("\n\nThis script tests the background offset calculation functionality with an NSI dataset.\n\n")
print("This script tests the background offset calculation functionality with an NSI dataset. The test includes: \
\n\t * downloading NSI dataset from specified urls;\
\n\t * Loading object scans, blank scan, dark scan, view angles, and conebeam geometry parameters;\
\n\t * Computing sinogram from object scan, blank scan, and dark scan images;\
\n\t * Performing background offset correction to the sinogram data;\
\n\t * Displaying a sinogram view before and after the background offset correction.\n")

# ###########################################################################
# Set the parameters to get the data 
# ###########################################################################

# ###################### Change the parameters below for your own use case.
# ##### params for dataset downloading.
# path to store output recon images
save_path = './output/test_background_offset/'
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
downsample_factor = [4, 4]
# ######### End of parameters #########

# ###########################################################################
# NSI preprocess: obtain sinogram, sino weights, angles, and geometry params
# ###########################################################################
obj_scan, blank_scan, dark_scan, angles, geo_params, defective_pixel_list = \
        mbircone.preprocess.NSI_load_scans_and_params(nsi_config_file_path, obj_scan_path,
                                                      blank_scan_path, dark_scan_path,
                                                      downsample_factor=downsample_factor,
                                                      defective_pixel_path=defective_pixel_path
                                                     )

print("MBIR geometry paramemters:")
pp.pprint(geo_params)
print('obj_scan shape = ', obj_scan.shape)
print('blank_scan shape = ', blank_scan.shape)
print('dark_scan shape = ', dark_scan.shape)

print("\n*******************************************************",
      "\n****** Computing sinogram data from scan images *******",
      "\n*******************************************************")
sino, defective_pixel_list = \
        mbircone.preprocess.transmission_CT_compute_sino(obj_scan, blank_scan, dark_scan,
                                                         defective_pixel_list
                                                        )
num_views, num_det_rows, num_det_channels = sino.shape
print('sino shape = ', sino.shape)

# delete scan images to optimize memory usage
del obj_scan, blank_scan, dark_scan

# ###########################################################################
# Background offset calculation
# ###########################################################################
print("\n*******************************************************",
      "\n*********** Calculating background offset *************",
      "\n*******************************************************")
edge_width = 9 # same as default value in calc_background_offset()
background_offset = mbircone.preprocess.calc_background_offset(sino, edge_width=edge_width)
print("Calculated background offset = ", background_offset)

sino_corrected = sino - background_offset

print("\n*************************************************************",
      "\n** Plotting sinogram with background information annotated **",
      "\n*************************************************************")
########## plot sinogram view with bacground information
plt.ion()
fig = plt.figure()
imgplot = plt.imshow(sino[0], vmin=-0.1, vmax=0.1, interpolation='none')
plt.title(label="Original sinogram view with background region information")
imgplot.set_cmap('gray')
plt.colorbar()

# overlay background region on top of the sinogram plot 
plt.axhline(y=edge_width, color="r") # top edge region
plt.axvline(x=edge_width, color="r") # left edge region
plt.axvline(x=num_det_channels-edge_width, color="r") # right edge region
plt.savefig(os.path.join(save_path, "sino_view_original.png"))

########## plot sinogram view after background offset correction
fig = plt.figure()
imgplot = plt.imshow(sino_corrected[0], vmin=-0.1, vmax=0.1, interpolation='none')
imgplot.set_cmap('gray')
plt.colorbar()
plt.title(label=f"Corrected sinogram view. Background offset = {background_offset:.6f}")
plt.savefig(os.path.join(save_path, "sino_view_corrected.png"))

########## Annotate the negative sinogram pixels
fig = plt.figure()
imgplot = plt.imshow(sino[0], vmin=-0.1, vmax=0.1, interpolation='none')
imgplot.set_cmap('gray')
plt.colorbar()
x_list, y_list = np.where(sino[0]<0)
plt.scatter(y_list, x_list, c='red', s=0.5)
#plt.scatter(0, 100, c='red')
plt.title(label="Original sinogram view with negative pixels annotated in red")
plt.savefig(os.path.join(save_path, 'sino_view_original_with_negative_pixels.png'))

num_negative_pixels = np.sum(sino<0)
perc_negative_pixels = 100*num_negative_pixels/np.prod(sino.shape) 
print(f"total number of negative pixels in original sinogram = {num_negative_pixels} ({perc_negative_pixels:.2f}% of total pixels)")

########## Annotate the negative sinogram pixels
fig = plt.figure()
imgplot = plt.imshow(sino_corrected[0], vmin=-0.1, vmax=0.1, interpolation='none')
imgplot.set_cmap('gray')
plt.colorbar()
x_list, y_list = np.where(sino_corrected[0]<0)
plt.scatter(y_list, x_list, c='red', s=0.5)
plt.title(label="Corrected sinogram view with negative pixels annotated in red")
plt.savefig(os.path.join(save_path, 'sino_view_corrected_with_negative_pixels.png'))

num_negative_pixels = np.sum(sino_corrected<0)
perc_negative_pixels = 100*num_negative_pixels/np.prod(sino_corrected.shape) 
print(f"total number of negative pixels in corrected sinogram = {num_negative_pixels} ({perc_negative_pixels:.2f}% of total pixels)")
input("press Enter")
