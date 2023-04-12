import os, sys
import numpy as np
import scipy
import math
import urllib.request
import tarfile
import mbircone
import demo_utils
import time
import pprint
pp = pprint.PrettyPrinter(indent=4)
from scipy.ndimage import zoom
"""
This script is a demonstration of the metal artifact reduction (MAR) functionality in preprocess module. For more information please read the `[theory] <theory.html>`_ section in readthedocs.

Demo functionality includes:
 * downloading NSI dataset from specified urls;
 * Computing sinogram from object scan, blank scan, and dark scan images;
 * Perform an intial MBIR reconstruction with "transmission" weight type;
 * Computing an adaptive data weight matrix from the sinogram and the initial recon. This weight will be used to reduce metal artifacts.
 * Performing an MBIR recon with reduced metal artifacts;
 * Displaying the results.
"""
print('This script is a demonstration of the metal artifact removal functionality in preprocess module. For more information please read the "theory" section in readthedocs.\
\nDemo functionality includes:\
\n\t * downloading NSI dataset from specified urls;\
\n\t * Computing sinogram from object scan, blank scan, and dark scan images;\
\n\t * Perform an initial MBIR reconstruction with "transmission" weight type;\
\n\t * Computing an adaptive data weight matrix from the sinogram and the initial recon. This weight will be used to reduce metal artifacts.\
\n\t * Performing an MBIR recon with reduced metal artifacts;\
\n\t * Displaying the results\n')

# ###########################################################################
# Set the parameters to get the data and do the recon 
# ###########################################################################

# ###################### Change the parameters below for your own use case.
# ##### params for dataset downloading.
# path to store output recon images
save_path = './output/mar_demo_background_offset_corrected_ds8_gamma4.0_beta1.0/'
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
downsample_factor = [8, 8]

# ##### parameters for background offset correction:
"""
the background offset is the average of sinogram pixels inside the box defined by parameters (x,y,width,height):
:                +------------------+
:                |                  |
:              height               |
:                |                  |
:               (xy)---- width -----+
"""
x_start = 0 # column index for the anchor point of background box
y_start = 0 # row index for the anchor point of background box
width = 192 # width of the background box
height = 15 # height of the background box

# ##### parameters for MBIR recon
# region of reconstruction (ROR) parameters. This should contain the object of interest.
num_image_rows = 160 # number of rows in reconstructed image. Set to None if unknown.
num_image_cols = 160 # number of columns in reconstructed image. Set to None if unknown.
num_image_slices = 200 # number of slices in reconstructed image. Set to None if unknown.
image_slice_offset = -3.0 # Vertical offset of the image in units of :math:`mm^{-1}`. Set to 0.0 if unknown.
# ##### parameters for MAR sinogram weight
# threshold value to identify metal voxels. Units: :math:`mm^{-1}`
metal_threshold = 0.1
# scalar values controlling MAR sino weights
# beta controls the weight to sinogram entries with low photon counts.
beta = 1.0
# gamma controls the weight to sinogram entries in which the projection paths contain metal components.
gamma = 4.0
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
sino, weights_transmission = mbircone.preprocess.transmission_CT_preprocess(obj_scan, blank_scan, dark_scan, weight_type='transmission', background_box_info = [(x_start,y_start,width,height),])
print('sino shape = ', sino.shape)

# delete scan images to reduce memory usage
del obj_scan, blank_scan, dark_scan

# ###########################################################################
# Perform MBIR reconstruction with "transmission" sino weight
# ###########################################################################
print("\n***************************************************************",
      "\n***** Performing MBIR recon with transmission sino weight *****",
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
recon_transmission = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification,
                                           det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                           delta_det_row=delta_det_row, delta_det_channel=delta_det_channel,
                                           weights=weights_transmission,
                                           num_image_rows=num_image_rows, num_image_cols=num_image_cols,
                                           num_image_slices=num_image_slices, image_slice_offset=image_slice_offset
                                         )
t_end = time.time()
t_elapsed = t_end - t_start
print(f"Recon with transmission sino weight finished. Time elapsed {t_elapsed:.1f} sec.")
np.save(os.path.join(save_path, "recon_transmission.npy"), recon_transmission)

# ###########################################################################
# Calculate MAR sinogram weight
# ###########################################################################
print("\n***************************************************************",
      "\n*************** Calculating MAR sinogram weight ***************",
      "\n***************************************************************")
weights_mar = mbircone.preprocess.calc_weight_mar(sino, init_recon=recon_transmission, 
                                                  angles=angles, dist_source_detector=dist_source_detector, magnification=magnification,
                                                  metal_threshold=metal_threshold, 
                                                  good_pixel_mask=weights_transmission,
                                                  beta=beta, gamma=gamma, 
                                                  delta_det_channel=delta_det_channel, delta_det_row=delta_det_row,
                                                  det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                                  image_slice_offset=image_slice_offset)
# delete transmission weight matrix to reduce memory usage
del weights_transmission

# ###########################################################################
# Perform MBIR recon with MAR sinogram weight
# ###########################################################################
print("\n***************************************************************",
      "\n******* Performing MBIR recon with MAR sinogram weight ********",
      "\n********* This step will take 30-60 minutes to finish *********",
      "\n***************************************************************")
t_start = time.time()
recon_mar = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification,
                                  det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                  delta_det_row=delta_det_row, delta_det_channel=delta_det_channel,
                                  weights=weights_mar,
                                  init_image=np.copy(recon_transmission), 
                                  num_image_rows=num_image_rows, num_image_cols=num_image_cols,
                                  num_image_slices=num_image_slices, image_slice_offset=image_slice_offset,
                                  #num_threads=18, 
                                 )
t_end = time.time()
t_elapsed = t_end - t_start
print(f"Recon with MAR weight finished. Time elapsed {t_elapsed:.1f} sec.")
np.save(os.path.join(save_path, 'recon_mar.npy'), recon_mar)
print("\n*******************************************************",
      "\n**************** Plotting recon slices ****************",
      "\n*******************************************************")
# rotate the recon images to an upright pose for display purpose
rot_angle = 17.165 # rotate angle in the plane defined by axes [0,2].
recon_transmission_transformed = scipy.ndimage.rotate(recon_transmission, rot_angle, [0,2], reshape=False, order=5)
recon_mar_transformed = scipy.ndimage.rotate(recon_mar, rot_angle, [0,2], reshape=False, order=5)
# axial slice
demo_utils.plot_image(recon_transmission_transformed[67], title=f'MBIR recon with "transmission" weight type',
                      filename=os.path.join(save_path, 'recon_transmission_axial67.png'), vmin=0, vmax=0.055)

demo_utils.plot_image(recon_mar_transformed[67], title=f'MBIR recon with "MAR" weight type',
                      filename=os.path.join(save_path, 'recon_mar_axial67.png'), vmin=0, vmax=0.055)

# sagittal slice
demo_utils.plot_image(recon_transmission_transformed[:,:,79], title=f'MBIR recon with "transmission" weight type',
                      filename=os.path.join(save_path, 'recon_transmission_sagittal79.png'), vmin=0, vmax=0.055)

demo_utils.plot_image(recon_mar_transformed[:,:,79], title=f'MBIR recon with "MAR" weight type',
                      filename=os.path.join(save_path, 'recon_mar_sagittal79.png'), vmin=0, vmax=0.055)

print(f"Recon images saved to {save_path}.")
input("press Enter")
