import os, sys
import numpy as np
import math
import urllib.request
import tarfile
import mbircone
import demo_utils, denoiser_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
This script is a demonstration of the qGGMRF denoiser. Demo functionality includes
 * downloading 3D ground truth image from the specified url
 * generating noisy image by adding Gaussian white noise to the ground truth
 * denoising the image and displaying them.
"""
print('This script is a demonstration of the qGGMRF denoiser. Demo functionality includes \
\n\t * downloading 3D ground truth image from the specified url \
\n\t * generating noisy image by adding Gaussian white noise to the ground truth \
\n\t * denoising the image and displaying them.')

# ###########################################################################
# Set the parameters to get the ground truth
# ###########################################################################
# url to download the 3D image volume clean_image file
clean_image_url = "https://engineering.purdue.edu/~bouman/data_repository/data/bottle_cap_3D_phantom.npy.tgz"  # name of the clean_image file. Please make sure this matches with the name of the downloaded file.
clean_image_name = 'bottle_cap_3D_phantom.npy'

# destination path to download and extract the clean_image file.
target_dir = './demo_data/'   

# path to store output recon images
save_path = './output/denoiser3D/'
os.makedirs(save_path, exist_ok=True)

# noise and denoiser parameters
sigma_w = 0.1              # Level of Gaussian noise to be added to clean_image
sharpness = 0.0            # Controls regularization of the denoiser: larger => sharper; smaller => smoother. Default is 0.0 in cone3D.denoise()
# ######### End of parameters #########

# ###########################################################################
# Download and extract the clean_image 
# ###########################################################################
# download clean_image file
clean_image_dir = demo_utils.download_and_extract(clean_image_url, target_dir)
clean_image_full_path = os.path.join(clean_image_dir, clean_image_name)
# load original clean_image
clean_image = np.load(clean_image_full_path)
print("shape of clean_image = ", clean_image.shape)

# ###########################################################################
# Generate noisy image by adding Gaussian noise to the ground truth
# ###########################################################################
print("Generating noisy image ...")
noise = np.random.normal(0, sigma_w, size=clean_image.shape)
noisy_image = clean_image + noise

# ###########################################################################
# Denoise
# ###########################################################################
print("Denoising ...")
denoised_image = mbircone.cone3D.denoise(noisy_image, sharpness=sharpness)

# ###########################################################################
# Generating plots to clean_image, noisy_image, and denoised_image ...
# ###########################################################################
print("Generating plots to clean_image, noisy_image, and denoised_image ...")
# Plot axial slices of clean_image, noisy_image, and denoised_image
display_slices = [7, 12, 17, 22]
for display_slice in display_slices:
    demo_utils.plot_image(clean_image[display_slice], title=f'clean_image, axial slice {display_slice}',
                          filename=os.path.join(save_path, f'clean_image_slice{display_slice}.png'), vmin=0, vmax=1.0)
    demo_utils.plot_image(noisy_image[display_slice], title=f'noisy_image, axial slice {display_slice}',
                          filename=os.path.join(save_path, f'noisy_image_slice{display_slice}.png'), vmin=0, vmax=1.0)
    demo_utils.plot_image(denoised_image[display_slice], title=f'denoised_image, axial slice {display_slice}',
                          filename=os.path.join(save_path, f'denoised_image_slice{display_slice}.png'), vmin=0, vmax=1.0)

input("press Enter")
