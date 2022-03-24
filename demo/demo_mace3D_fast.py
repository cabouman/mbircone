import os, sys
import numpy as np
import math
import urllib.request
import tarfile
from keras.models import model_from_json
import mbircone
import demo_utils, denoiser_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
This script is a quick demonstration of the mace3D reconstruction algorithm.  Demo functionality includes
 * downloading phantom and denoiser data from specified urls
 * downsampling the phantom along all three dimensions
 * generating sinogram by projecting the phantom and then adding transmission noise
 * performing a 3D MACE reconstruction.
"""
print('This script is a quick demonstration of the mace3D reconstruction algorithm.  Demo functionality includes \
\n\t * downloading phantom and denoiser data from specified urls \
\n\t * downsampling the phantom along all three dimensions \
\n\t * generating sinogram by projecting the phantom and then adding transmission noise\
\n\t * performing a 3D MACE reconstruction.')

# ###########################################################################
# Set the parameters to get the data and do the recon 
# ###########################################################################

# Change the parameters below for your own use case.

# Denoiser function to be used in MACE. For the built-in demo, this should be one of dncnn_keras or dncnn_ct
# Other denoisers built in keras can be used with minimal modification by putting the architecture and weights
# in model.json and model.hdf5 in the denoiser_path set below
denoiser_type = 'dncnn_ct'

# The url to the data repo.
data_repo_url = 'https://github.com/cabouman/mbir_data/raw/master/'

# Download url to the index file.
# This file will be used to retrieve urls to files that we are going to download
yaml_url = os.path.join(data_repo_url, 'index.yaml')

# Choice of phantom and denoiser files. 
# These should be valid choices specified in the index file. 
# The urls to phantom data and NN weights will be parsed from data_repo_url and the choices of phantom and denoiser specified below.
phantom_name = 'bottle_cap_3D'
denoiser_name = denoiser_type

# destination path to download and extract the phantom and NN weight files.
target_dir = './demo_data/'   
# path to store output recon images
save_path = './output/mace3D_fast/'  

# Geometry parameters
dist_source_detector = 839.0472     # Distance between the X-ray source and the detector in units of ALU
magnification = 5.572128439964856   # magnification = (source to detector distance)/(source to center-of-rotation distance)
delta_pixel_detector = 0.25         # Scalar value of detector pixel spacing in units of ALU
num_det_rows = 29                   # number of detector rows
num_det_channels = 120              # number of detector channels

# Simulated sinogram parameters
num_views = 50               # number of projection views
sino_noise_sigma = 0.005      # transmission noise level

# MACE recon parameters
max_admm_itr = 10            # max ADMM iterations for MACE reconstruction
sharpness = 1.0
prior_weight = 0.5           # cumulative weights for three prior agents.
# ######### End of parameters #########


# ###########################################################################
# Download and extract data 
# ###########################################################################

# Download the url index file and return path to local file. 
index_path = demo_utils.download_and_extract(yaml_url, target_dir) 
# Load the url index file as a directionary
url_index = demo_utils.load_yaml(index_path)
# get urls to phantom and denoiser parameter file
phantom_url = os.path.join(data_repo_url, url_index['phantom'][phantom_name])  # url to download the 3D image volume phantom file
denoiser_url = os.path.join(data_repo_url, url_index['denoiser'][denoiser_name])  # url to download the denoiser parameter file 

# download phantom file
phantom_path = demo_utils.download_and_extract(phantom_url, target_dir)
# download and extract NN weights and structure files
denoiser_path = demo_utils.download_and_extract(denoiser_url, target_dir)


# ###########################################################################
# Generate downsampled phantom 
# ###########################################################################
print("Generating downsampled 3D phantom volume ...")

# load original phantom
phantom_orig = np.load(phantom_path)
print("shape of original phantom = ", phantom_orig.shape)

# downsample the original phantom along slice axis
(Nz, Nx, Ny) = phantom_orig.shape
Nx_ds = Nx // 2 + 1
Ny_ds = Ny // 2 + 1
phantom = demo_utils.image_resize(phantom_orig, (Nx_ds, Ny_ds))
print("shape of downsampled phantom = ", phantom.shape)


# ###########################################################################
# Generate sinogram
# ###########################################################################
print("Generating sinogram ...")

# Generate view angles and sinogram with weights
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
sino = mbircone.cone3D.project(phantom, angles,
                               num_det_rows, num_det_channels,
                               dist_source_detector, magnification,
                               delta_pixel_detector=delta_pixel_detector)
weights = mbircone.cone3D.calc_weights(sino, weight_type='transmission')

# Add transmission noise
noise = sino_noise_sigma * 1. / np.sqrt(weights) * np.random.normal(size=(num_views, num_det_rows, num_det_channels))
sino_noisy = sino + noise


# ###########################################################################
# Set up the denoiser
# ###########################################################################
# This demo includes a custom CNN trained on CT images and a generic DnCNN in keras 
# The choice is set in the parameter section above.
print("Loading denoiser function and model ...")

# DnCNN denoiser trained on CT images. This is the denoiser that we recommend using.
if denoiser_type == 'dncnn_ct':
    print("Denoiser function: custom DnCNN trained on CT images.")

    # Load denoiser model structure and weights
    denoiser_model = denoiser_utils.DenoiserCT(checkpoint_dir=os.path.join(denoiser_path, 'model_dncnn_ct'))

    # Define the denoiser using this model.  This version requires some interface code to match with MACE.
    def denoiser(img_noisy):
        img_noisy = np.expand_dims(img_noisy, axis=1)
        upper_range = denoiser_utils.calc_upper_range(img_noisy)
        img_noisy = img_noisy/upper_range
        testData_obj = denoiser_utils.DataLoader(img_noisy)
        img_denoised = denoiser_model.denoise(testData_obj, batch_size=256)
        img_denoised = img_denoised*upper_range
        return np.squeeze(img_denoised)
# DnCNN denoiser in Keras. This denoiser model is trained on natural images. 
elif denoiser_type == 'dncnn_keras':
    print("Denoiser function: DnCNN trained on natural images.")

    # Load denoiser model structure and weights
    json_path = os.path.join(denoiser_path, 'model_dncnn_keras/model.json')  # model architecture file
    weight_path = os.path.join(denoiser_path, 'model_dncnn_keras/model.hdf5')  # model weight file
    with open(json_path, 'r') as json_file:
        denoiser_model = model_from_json(json_file.read())  # load model architecture

    denoiser_model.load_weights(weight_path)  # load model weights

    # Define the denoiser using this model.
    def denoiser(img_noisy):
        img_noisy = img_noisy[..., np.newaxis]  # (Nz,N0,N1,...,Nm,1)
        img_denoised = denoiser_model.predict(img_noisy)  # inference
        return np.squeeze(img_denoised)
else:
    raise RuntimeError('Unkown denoiser_type. Should be either dncnn_ct or dncnn_keras.')


# ###########################################################################
# Perform MACE reconstruction
# ###########################################################################
print("Performing MACE reconstruction ...")
recon_mace = mbircone.mace.mace3D(sino_noisy, angles, dist_source_detector, magnification,
                                  denoiser=denoiser, denoiser_args=(),
                                  max_admm_itr=max_admm_itr, prior_weight=prior_weight,
                                  sharpness=sharpness,
                                  delta_pixel_detector=delta_pixel_detector,
                                  weight_type='transmission',
                                  save_path=save_path)
recon_shape = recon_mace.shape
print("Reconstruction shape = ", recon_shape)


# ###########################################################################
# Post-process reconstruction results
# ###########################################################################
print("Post processing MACE reconstruction results ...")
# Save recon results as a numpy array
os.makedirs(save_path, exist_ok=True)
np.save(os.path.join(save_path, "recon_mace.npy"), recon_mace)
# load qGGMRF recon
recon_qGGMRF = np.load(os.path.join(save_path, "recon_qGGMRF.npy"))

# Plot axial slices of phantom and recon
display_slices = [7, 10, 13, 16, 19, 22]
for display_slice in display_slices:
    demo_utils.plot_image(phantom[display_slice], title=f'phantom, axial slice {display_slice}',
                          filename=os.path.join(save_path, f'phantom_slice{display_slice}.png'), vmin=0, vmax=0.5)
    demo_utils.plot_image(recon_mace[display_slice], title=f'MACE reconstruction, axial slice {display_slice}',
                          filename=os.path.join(save_path, f'recon_mace_slice{display_slice}.png'), vmin=0, vmax=0.5)
    demo_utils.plot_image(recon_qGGMRF[display_slice], title=f'qGGMRF reconstruction, axial slice {display_slice}',
                          filename=os.path.join(save_path, f'recon_qGGMRF_slice{display_slice}.png'), vmin=0, vmax=0.5)
# Plot 3D phantom and recon image volumes as gif images.
demo_utils.plot_gif(phantom, save_path, 'phantom_resized', vmin=0, vmax=0.5)
demo_utils.plot_gif(recon_mace, save_path, 'recon_mace', vmin=0, vmax=0.5)
demo_utils.plot_gif(recon_qGGMRF, save_path, 'recon_qGGMRF', vmin=0, vmax=0.5)

input("press Enter")
