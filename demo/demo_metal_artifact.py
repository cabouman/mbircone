import os, sys
import numpy as np
import scipy
import math
import urllib.request
import tarfile
import mbircone
import demo_utils
import pprint
import time
pp = pprint.PrettyPrinter(indent=4)

"""
This script is a demonstration of the metal artifact reduction (MAR) functionality using MAR sinogram weight. For more information please refer to the `[Theory] <theory.html>`_ section in readthedocs.
Demo functionality includes:
 * downloading NSI dataset from specified urls;
 * Computing sinogram data;
 * Computing two sets of sinogram weights, one with type "transmission" and the other with type "MAR";
 * Computing two sets of MBIR reconstructions with different sinogram weights respectively;
 * Displaying the results.
"""

print('This script is a demonstration of the metal artifact reduction (MAR) functionality using MAR sinogram weight. For more information please refer to the Theory section in readthedocs. \
\n Demo functionality includes:\
\n\t * downloading NSI dataset from specified urls;\
\n\t * Computing sinogram data;\
\n\t * Computing two sets of sinogram weights, one with type "transmission" and the other with type "MAR";\
\n\t * Computing two sets of MBIR reconstructions with each sinogram weight respectively;\
\n\t * Displaying the results.\n')
# ###########################################################################
# Set the parameters to get the data and do the recon 
# ###########################################################################

# ###################### User defined params. Change the parameters below for your own use case.
save_path = './output/demo_metal_artifacts/' # path to store output recon images
os.makedirs(save_path, exist_ok=True) # mkdir if directory does not exist
downsample_factor = [6, 6] # downsample factor of scan images along detector rows and detector columns.
# view subsample factor
subsample_view_factor = 2

# ##### parameters for MAR sinogram weight
# threshold value to identify metal voxels. 
metal_threshold = 0.1 #Units: :math:`mm^{-1}`
# beta controls the weight to sinogram entries with low photon counts.
# A larger beta promotes image homogeneity.
beta = 1.0
# gamma controls the weight to sinogram entries in which the projection paths contain metal components.
# A larger gamma promotes metal artifacts reduction around metal regions.
gamma = 4.0

# ##### params for dataset downloading. User may change these parameters for their own datasets.
# An example NSI dataset will be downloaded from `dataset_url`, and saved to `download_dir`.
# url to NSI dataset.
dataset_url = 'https://engineering.purdue.edu/~bouman/data_repository/data/mar_demo_data.tgz'
# destination path to download and extract the NSI data and metadata.
download_dir = './demo_data/'
# download dataset. The download path will be later used to define path to dataset specific files.
download_dir = demo_utils.download_and_extract(dataset_url, download_dir)
# path to NSI dataset
dataset_path = os.path.join(download_dir, "mar_demo_data") # change this for different NSI datasets.

# ######### NSI specific file paths, These are derived from dataset_path.
# User may change the variables below for a different NSI dataset.
# path to NSI config file. Change dataset path params for your own NSI dataset
nsi_config_file_path = os.path.join(dataset_path, 'JB-033_ArtifactPhantom_VerticalMetal.nsipro')
# path to "Geometry Report.rtf"
geom_report_path = os.path.join(dataset_path, 'Geometry_Report_nsi_demo.rtf')
# path to directory containing all object scans
obj_scan_path = os.path.join(dataset_path, 'Radiographs-JB-033_ArtifactPhantom_VerticalMetal')
# path to blank scan. Usually <dataset_path>/Corrections/gain0.tif
blank_scan_path = os.path.join(dataset_path, 'Corrections/gain0.tif')
# path to dark scan. Usually <dataset_path>/Corrections/offset.tif
dark_scan_path = os.path.join(dataset_path, 'Corrections/offset.tif')
# path to NSI file containing defective pixel information
defective_pixel_path = os.path.join(dataset_path, 'Corrections/defective_pixels.defect')
# ###################### End of parameters

t_start = time.time()
# ###########################################################################
# NSI preprocess: obtain sinogram, sino weights, angles, and geometry params
# ###########################################################################
print("\n********************************************************************************",
      "\n** Load scan images, angles, geometry params, and defective pixel information **",
      "\n********************************************************************************")
obj_scan, blank_scan, dark_scan, angles, geo_params, defective_pixel_list = \
        mbircone.preprocess.NSI_load_scans_and_params(nsi_config_file_path, geom_report_path,
                                                      obj_scan_path, blank_scan_path, dark_scan_path,
                                                      defective_pixel_path,
                                                      downsample_factor=downsample_factor,
                                                      subsample_view_factor=subsample_view_factor)
print("MBIR geometry paramemters:")
pp.pprint(geo_params)
print('obj_scan shape = ', obj_scan.shape)
print('blank_scan shape = ', blank_scan.shape)
print('dark_scan shape = ', dark_scan.shape)

print("\n*******************************************************",
      "\n********** Compute sinogram from scan images **********",
      "\n*******************************************************")
sino, defective_pixel_list = \
        mbircone.preprocess.transmission_CT_compute_sino(obj_scan, blank_scan, dark_scan,
                                                         defective_pixel_list
                                                        )
# delete scan images to optimize memory usage
del obj_scan, blank_scan, dark_scan

print("\n*******************************************************",
      "\n********* Interpolate defective sino entries **********",
      "\n*******************************************************")
sino, defective_pixel_list = mbircone.preprocess.interpolate_defective_pixels(sino, defective_pixel_list)

print("\n*******************************************************",
      "\n************** Correct background offset **************",
      "\n*******************************************************")
background_offset = mbircone.preprocess.calc_background_offset(sino)
print("background_offset = ", background_offset)
sino = sino - background_offset

print("\n*******************************************************",
      "\n**** Rotate sino images w.r.t. rotation axis tilt *****",
      "\n*******************************************************")
sino = mbircone.preprocess.correct_tilt(sino, tilt_angle=geo_params["rotation_axis_tilt"])

print("\n*******************************************************",
      "\n******** Calculate transission sinogram weight ********",
      "\n*******************************************************")
weights_trans = mbircone.preprocess.calc_weights(sino, weight_type="transmission",
                                                 defective_pixel_list=defective_pixel_list
                                                )

# extract mbircone geometry params required for recon
dist_source_detector = geo_params["dist_source_detector"]
magnification = geo_params["magnification"]
delta_det_row = geo_params["delta_det_row"]
delta_det_channel = geo_params["delta_det_channel"]
det_channel_offset = geo_params["det_channel_offset"]
det_row_offset = geo_params["det_row_offset"]
rotation_offset = geo_params["rotation_offset"]

# ###########################################################################
# Perform MBIR reconstruction with "transmission" sino weight
# ###########################################################################
print("\n***************************************************************",
      "\n***** Performing MBIR recon with transmission sino weight *****",
      "\n********* This step will take 15-30 minutes to finish *********",
      "\n***************************************************************")
print("This recon will be used to identify metal voxels and compute the MAR sinogram weight.")
# MBIR recon
recon_trans = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification,
                                    det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                    rotation_offset=rotation_offset,
                                    delta_det_row=delta_det_row, delta_det_channel=delta_det_channel,
                                    weights=weights_trans)
np.save(os.path.join(save_path, "recon_trans.npy"), recon_trans)

# ###########################################################################
# Calculate MAR sinogram weight
# ###########################################################################
print("\n***************************************************************",
      "\n*************** Calculating MAR sinogram weight ***************",
      "\n***************************************************************")
weights_mar = mbircone.preprocess.calc_weights_mar(sino, angles=angles, init_recon=recon_trans,
                                                   dist_source_detector=dist_source_detector, magnification=magnification,
                                                   metal_threshold=metal_threshold,
                                                   beta=beta, gamma=gamma,
                                                   defective_pixel_list=defective_pixel_list,
                                                   delta_det_channel=delta_det_channel, delta_det_row=delta_det_row,
                                                   det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                                   rotation_offset=rotation_offset,
                                                   )
# delete transmission weight matrix to reduce memory usage
del weights_trans

print("\n***************************************************************",
      "\n********* Performing MBIR recon with MAR sino weight **********",
      "\n********* This step will take 30-60 minutes to finish *********",
      "\n***************************************************************")

print("This recon will be used to identify metal voxels and compute the MAR sinogram weight.")
# MBIR recon
recon_mar = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification,
                                  det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                  rotation_offset=rotation_offset,
                                  delta_det_row=delta_det_row, delta_det_channel=delta_det_channel,
                                  weights=weights_mar,
                                  init_image=recon_trans,
                                  max_resolutions=2
                                 )
np.save(os.path.join(save_path, "recon_mar.npy"), recon_mar)


print("MBIR recon finished. recon shape = ", np.shape(recon_mar))

print("\n*******************************************************",
      "\n****************** Plot recon slices ******************",
      "\n*******************************************************")
# rotate the recon images to an upright pose for display purpose
rot_angle = 17.165 # rotate angle in the plane defined by axes [0,2].
recon_trans_transformed = scipy.ndimage.rotate(recon_trans, rot_angle, [0,2], reshape=False, order=5)
recon_mar_transformed = scipy.ndimage.rotate(recon_mar, rot_angle, [0,2], reshape=False, order=5)
# axial slice
demo_utils.plot_image(recon_trans_transformed[135], title=f'MBIR recon with "transmission" weight type',
                      filename=os.path.join(save_path, 'recon_trans_axial135.png'), vmin=0, vmax=0.055)

demo_utils.plot_image(recon_mar_transformed[135], title=f'MBIR recon with "MAR" weight type',
                      filename=os.path.join(save_path, 'recon_mar_axial135.png'), vmin=0, vmax=0.055)

# sagittal slice
demo_utils.plot_image(recon_trans_transformed[:,:,130], title=f'MBIR recon with "transmission" weight type',
                      filename=os.path.join(save_path, 'recon_trans_sagittal130.png'), vmin=0, vmax=0.055)

demo_utils.plot_image(recon_mar_transformed[:,:,130], title=f'MBIR recon with "MAR" weight type',
                      filename=os.path.join(save_path, 'recon_mar_sagittal130.png'), vmin=0, vmax=0.055)

t_end = time.time()
t_elapsed = t_end - t_start
print(f"Recon images saved to {save_path}. Demo script takes {t_elapsed:.1f} sec to finish.")
input("press Enter")
