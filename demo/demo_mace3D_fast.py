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
 * downloading phantom and denoiser data from a url
 * downsampling the phantom along all three dimensions
 * generating sinogram by projecting the phantom
 * performing a 3D MACE reconstruction.
"""
print('This script is a quick demonstration of the mace3D reconstruction algorithm.  Demo functionality includes \
\n\t * downloading phantom and denoiser data from a url \
\n\t * downsampling the phantom along all three dimensions \
\n\t * generating sinogram by projecting the phantom \
\n\t * performing a 3D MACE reconstruction.')

# ######### Set the parameters to get the data and do the recon #########
# Change the parameters below for your own use case.

# Denoiser function to be used in MACE. For the built-in demo, this should be one of dncnn_keras or dncnn_ct
# Other denoisers built in keras can be used with minimal modification by putting the architecture and weights
# in model.json and model.hdf5 in the model_param_path set below
denoiser_type = 'dncnn_ct'

# The url to phantom data and NN weights and set the local extract path.
download_url = 'https://github.com/dyang37/mbircone_data/raw/master/demo_data.tar.gz'  # url to download the demo data
extract_path = './demo_data/'   # destination path to extract the downloaded tarball file

# Path to downloaded files. Please change them accordingly if you replace any of them with your own files.
model_param_path = os.path.join(extract_path, './dncnn_params/')  # pre-trained dncnn model parameter files
phantom_path = os.path.join(extract_path, 'phantom_3D.npy')  # 3D image volume phantom file
output_dir = './output/mace3D_fast/'  # path to store output recon images

# Geometry parameters
dist_source_detector = 839.0472     # Distance between the X-ray source and the detector in units of ALU
magnification = 5.572128439964856   # magnification = (source to detector distance)/(source to center-of-rotation distance)
delta_pixel_detector = 0.25         # Scalar value of detector pixel spacing in units of ALU
num_det_rows = 14                   # number of detector rows
num_det_channels = 120              # number of detector channels

# Simulated sinogram parameters
num_views = 75               # number of projection views
sino_noise_sigma = 0.01      # transmission noise level

# MACE recon parameters
max_admm_itr = 10            # max ADMM iterations for MACE reconstruction
prior_weight = 0.5           # cumulative weights for three prior agents.

# ######### End of parameters #########

# ######### Download and extract data #########
# A tarball file will be downloaded from the given url and extracted to extract_path.
# The tarball file downloaded from the default url in this demo contains the following files:
#   phantom_3D.npy:  an image volume phantom file. You can replace this file with your own phantom data.
#   dncnn_params/ directory:  dncnn parameter files
demo_utils.download_and_extract(download_url, extract_path)

# ######### Generate downsampled phantom #########
print("Generating downsampled 3D phantom volume ...")
# load original phantom
phantom_orig = np.load(phantom_path)
print("shape of original phantom = ", phantom_orig.shape)

# downsample the original phantom along slice axis
(Nz, Nx, Ny) = phantom_orig.shape
Nx_ds = Nx // 2 + 1
Ny_ds = Ny // 2 + 1
Nz_ds = Nz // 2
phantom = demo_utils.image_resize(phantom_orig, (Nx_ds, Ny_ds))

# Take first half of the slices to form the downsampled phantom.
phantom = phantom[:Nz_ds]
print("shape of downsampled phantom = ", phantom.shape)

# ######### Generate sinogram #########
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

# ######### Set up the denoiser #########
# This demo includes a generic DnCNN in keras and a custom CNN trained on CT images - the choice is set in
# the parameter section above.
print("Loading denoiser function and model ...")
if denoiser_type == "dncnn_keras":
    print("Denoiser function: DnCNN trained on natural images.")

    # Load denoiser model structure and weights
    json_path = os.path.join(model_param_path, 'model_dncnn/model.json')  # model architecture file
    weight_path = os.path.join(model_param_path, 'model_dncnn/model.hdf5')  # model weight file
    with open(json_path, 'r') as json_file:
        denoiser_model = model_from_json(json_file.read())  # load model architecture

    denoiser_model.load_weights(weight_path)  # load model weights

    # Define the denoiser using this model.
    def denoiser(img_noisy):
        img_noisy = img_noisy[..., np.newaxis]  # (Nz,N0,N1,...,Nm,1)
        img_denoised = denoiser_model.predict(img_noisy)  # inference
        return np.squeeze(img_denoised)

else:
    print("Denoiser function: custom DnCNN trained on CT images.")

    # Load denoiser model structure and weights
    denoiser_model_path = os.path.join(model_param_path, 'model_dncnn_video')
    denoiser_model = denoiser_utils.DenoiserCT(checkpoint_dir=denoiser_model_path)

    # Define the denoiser using this model.  This version requires some interface code to match with MACE.
    def denoiser(img_noisy):
        img_noisy = np.expand_dims(img_noisy, axis=1)
        testData_obj = denoiser_utils.DataLoader(img_noisy)
        denoiser_model.test(testData_obj)
        img_denoised = np.stack(testData_obj.outData, axis=0)
        return np.squeeze(img_denoised)

# ######### Perform MACE reconstruction #########
print("Performing MACE reconstruction ...")
recon_mace = mbircone.mace.mace3D(sino_noisy, angles, dist_source_detector, magnification,
                                  denoiser=denoiser, denoiser_args=(),
                                  max_admm_itr=max_admm_itr, prior_weight=prior_weight,
                                  delta_pixel_detector=delta_pixel_detector,
                                  weight_type='transmission')
recon_shape = recon_mace.shape
print("Reconstruction shape = ", recon_shape)

# ######### Post-process Reconstruction results #########
print("Post processing MACE reconstruction results ...")
# Save recon results as a numpy array
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "recon_mace.npy"), recon_mace)

# Plot sinogram data
demo_utils.plot_image(sino_noisy[0], title='sinogram view 0, noise level=0.05',
                      filename=os.path.join(output_dir, 'sino_noisy.png'), vmin=0, vmax=4)
demo_utils.plot_image(sino[0], title='clean sinogram view 0', filename=os.path.join(output_dir, 'sino_clean.png'),
                      vmin=0, vmax=4)
demo_utils.plot_image(noise[0], title='sinogram additive Gaussian noise,  view 0',
                      filename=os.path.join(output_dir, 'sino_transmission_noise.png'), vmin=-0.08, vmax=0.08)

# Plot axial slices of phantom and recon
display_slices = [2, recon_shape[0] // 2]
for display_slice in display_slices:
    demo_utils.plot_image(phantom[display_slice], title=f'phantom, axial slice {display_slice}',
                          filename=os.path.join(output_dir, f'phantom_slice{display_slice}.png'), vmin=0, vmax=0.5)
    demo_utils.plot_image(recon_mace[display_slice], title=f'MACE reconstruction, axial slice {display_slice}',
                          filename=os.path.join(output_dir, f'recon_mace_slice{display_slice}.png'), vmin=0, vmax=0.5)

# Plot 3D phantom and recon image volumes as gif images.
demo_utils.plot_gif(phantom_orig, output_dir, 'phantom_original', vmin=0, vmax=0.5)
demo_utils.plot_gif(phantom, output_dir, 'phantom_resized', vmin=0, vmax=0.5)
demo_utils.plot_gif(recon_mace, output_dir, 'recon_mace', vmin=0, vmax=0.5)

input("press Enter")
