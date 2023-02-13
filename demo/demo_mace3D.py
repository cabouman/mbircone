import os, sys
import numpy as np
import math
import urllib.request
import tarfile
import mbircone
import demo_utils, denoiser_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
This script is a demonstration of the mace3D reconstruction algorithm. Demo functionality includes
 * downloading phantom and denoiser data from specified urls
 * generating synthetic sinogram data by forward projecting the phantom
 * performing a 3D qGGMRF and MACE reconstructions and displaying them.
"""
print('This script is a demonstration of the mace3D reconstruction algorithm. Demo functionality includes \
\n\t * downloading phantom and denoiser data from specified urls \
\n\t * generating synthetic sinogram data by forward projecting the phantom\
\n\t * performing a 3D qGGMRF and MACE reconstructions and displaying them.')

# ###########################################################################
# Set the parameters to get the data and do the recon 
# ###########################################################################

# Change the parameters below for your own use case.

##### urls to phantom and denoiser parameter file
# url to download the 3D image volume phantom file
phantom_url = "https://engineering.purdue.edu/~bouman/data_repository/data/bottle_cap_3D_phantom.npy.tgz"  # name of the phantom file. Please make sure this matches with the name of the downloaded file.
phantom_name = 'bottle_cap_3D_phantom.npy'

# url to download the denoiser parameter file
denoiser_url = "https://engineering.purdue.edu/~bouman/data_repository/data/model_dncnn_ct.tgz"
# name of the directory containing all denoiser param files. Make sure this matches with the name of the downloaded file.
denoiser_model_name = 'model_dncnn_ct'


# destination path to download and extract the phantom and NN weight files.
target_dir = './demo_data/'   
# path to store output recon images
save_path = './output/mace3D/'
os.makedirs(save_path, exist_ok=True)

# Geometry parameters
dist_source_detector = 3356.1888    # Distance between the X-ray source and the detector in units of ALU
magnification = 5.572128439964856   # magnification = (source to detector distance)/(source to center-of-rotation distance)
num_det_rows = 28                   # number of detector rows
num_det_channels = 240              # number of detector channels
num_views = 75               # number of projection views

# MACE recon parameters
sharpness = 1.0              # Parameter to control regularization level of reconstruction.
max_admm_itr = 10            # max ADMM iterations for MACE reconstruction
# ######### End of parameters #########


# ###########################################################################
# Download and extract data 
# ###########################################################################

#### download phantom file
# retrieve parent directory where phantom npy file is extracted to
phantom_dir = demo_utils.download_and_extract(phantom_url, target_dir)
phantom_full_path = os.path.join(phantom_dir, phantom_name)

#### download and extract NN weights and structure files
# retrieve parent directory where denoiser files are extracted to
denoiser_model_dir = demo_utils.download_and_extract(denoiser_url, target_dir)
denoiser_model_full_path = os.path.join(denoiser_model_dir, denoiser_model_name)

# load original phantom
phantom = np.load(phantom_full_path) / 4.
print("shape of phantom = ", phantom.shape)


# ###########################################################################
# Generate sinogram
# ###########################################################################
print("Generating sinogram ...")

# Generate view angles and sinogram with weights
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
sino = mbircone.cone3D.project(phantom, angles,
                               num_det_rows, num_det_channels,
                               dist_source_detector, magnification)
# ###########################################################################
# Perform qGGMRF reconstruction
# ###########################################################################
print("Performing qGGMRF reconstruction. \
\nThis will be used as the initial image for mace3D reconstruction.")
recon_qGGMRF = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification,
                                     sharpness=sharpness,
                                     verbose=1)

# ###########################################################################
# Set up the denoiser
# ###########################################################################
# The MACE reconstruction algorithm requires the use of a denoiser as a prior model.
# This demo includes a custom DnCNN denoiser trained on CT images, which can be used for other applications.
print("Loading denoiser function and model ...")
# Load denoiser model structure and pre-trained model weights
denoiser_model = denoiser_utils.DenoiserCT(checkpoint_dir=denoiser_model_full_path)
# Define the denoiser using this model.  This version requires some interface code to match with MACE.
def denoiser(img_noisy):
    img_noisy = np.expand_dims(img_noisy, axis=1)
    upper_range = denoiser_utils.calc_upper_range(img_noisy)
    img_noisy = img_noisy/upper_range
    testData_obj = denoiser_utils.DataLoader(img_noisy)
    img_denoised = denoiser_model.denoise(testData_obj, batch_size=256)
    img_denoised = img_denoised*upper_range
    return np.squeeze(img_denoised)


# ###########################################################################
# Perform MACE reconstruction
# ###########################################################################
print("Performing MACE reconstruction ...")
recon_mace = mbircone.mace.mace3D(sino, angles, dist_source_detector, magnification,
                                  denoiser=denoiser, denoiser_args=(),
                                  max_admm_itr=max_admm_itr,
                                  init_image=recon_qGGMRF,
                                  sharpness=sharpness,
                                  verbose=1)
recon_shape = recon_mace.shape
print("Reconstruction shape = ", recon_shape)


# ###########################################################################
# Generating phantom and reconstruction images
# ###########################################################################
print("Generating phantom, sinogram, and reconstruction images ...")
# Plot sinogram views
for view_idx in [0, num_views//4, num_views//2]:
    demo_utils.plot_image(sino[view_idx], title=f'sinogram view {view_idx}', vmin=0, vmax=4.0,
                          filename=os.path.join(save_path, f'sino_view{view_idx}.png'))
# Plot axial slices of phantom and recon
display_slices = [7, 12, 17, 22]
for display_slice in display_slices:
    demo_utils.plot_image(phantom[display_slice], title=f'phantom, axial slice {display_slice}',
                          filename=os.path.join(save_path, f'phantom_slice{display_slice}.png'), vmin=0, vmax=0.2)
    demo_utils.plot_image(recon_mace[display_slice], title=f'MACE reconstruction, axial slice {display_slice}',
                          filename=os.path.join(save_path, f'recon_mace_slice{display_slice}.png'), vmin=0, vmax=0.2)
    demo_utils.plot_image(recon_qGGMRF[display_slice], title=f'qGGMRF reconstruction, axial slice {display_slice}',
                          filename=os.path.join(save_path, f'recon_qGGMRF_slice{display_slice}.png'), vmin=0, vmax=0.2)

input("press Enter")
