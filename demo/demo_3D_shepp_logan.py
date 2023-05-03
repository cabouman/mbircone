import os
import numpy as np
import mbircone
from demo_utils import plot_image, plot_gif

"""
This script is a demonstration of the 3D conebeam reconstruction algorithm with a qGGMRF prior. 
Demo functionality includes:
 * Generating a 3D Shepp Logan phantom;
 * Forward projecting the Shepp Logan phantom to form a synthetic sinogram;
 * Computing a 3D reconstruction from the sinogram using a qGGMRF prior model;
 * Displaying the results.
"""
print('This script is a demonstration of the 3D conebeam reconstruction algorithm with a qGGMRF prior.\
Demo functionality includes:\
\n\t * Generating a 3D Shepp Logan phantom; \
\n\t * Forward projecting the Shepp Logan phantom to form a synthetic sinogram;\
\n\t * Computing a 3D reconstruction from the sinogram using a qGGMRF prior model;\
\n\t * Displaying the results.\n')


# ###########################################################################
# Set the parameters to generate the phantom, synthetic sinogram, and do the recon
# ###########################################################################

# Change the parameters below for your own use case.

# Detector and geometry parameters
num_det_rows = 128                           # Number of detector rows
num_det_channels = 128                       # Number of detector channels
magnification = 2.0                          # Ratio of (source to detector)/(source to center of rotation)
dist_source_detector = 3.0*num_det_channels  # Distance from source to detector in ALU
num_views = 64                               # Number of projection views

# Generate uniformly spaced view angles in the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

# Set reconstruction parameters
sharpness = 0.0                             # Controls regularization: larger => sharper; smaller => smoother
T = 0.1                                     # Controls edginess of reconstruction

# Set phantom generation parameters
num_phantom_slices = num_det_rows           # Set number of phantom slices = to the number of detector rows
num_phantom_rows = num_det_channels         # Make number of phantom rows and columns = to number of detector columns
num_phantom_cols = num_det_channels

# Calculate scaling factor for Shepp Logan phantom so that projections are physically realistic -log attenuation values
SL_phantom_density_scale = 4.0*magnification/num_phantom_rows

# Set display parameters for Shepp Logan phantom
vmin = SL_phantom_density_scale*1.0
vmax = SL_phantom_density_scale*1.2

# local path to save phantom, sinogram, and reconstruction images
save_path = f'output/3D_shepp_logan/'
os.makedirs(save_path, exist_ok=True)

print('Genrating 3D Shepp Logan phantom ...\n')
######################################################################################
# Generate a 3D shepp logan phantom
######################################################################################
phantom = mbircone.phantom.gen_shepp_logan_3d(num_phantom_rows, num_phantom_cols, num_phantom_slices)
phantom = SL_phantom_density_scale*phantom
print('Phantom shape = ', np.shape(phantom))

######################################################################################
# Generate synthetic sinogram
######################################################################################
print('Generating synthetic sinogram ...\n')
sino = mbircone.cone3D.project(phantom, angles,
                               num_det_rows, num_det_channels,
                               dist_source_detector, magnification)
print('Synthetic sinogram shape: (num_views, num_det_rows, num_det_channels) = ', sino.shape)

######################################################################################
# Perform 3D MBIR reconstruction using qGGMRF prior
######################################################################################
print('Performing 3D qGGMRF reconstruction ...\n')
recon = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification, sharpness=sharpness, T=T)
(num_image_slices, num_image_rows, num_image_cols) = np.shape(recon)
print('recon shape = ', np.shape(recon))

######################################################################################
# Display phantom, synthetic sinogram, and reconstruction images
######################################################################################
# Set display indexes for phantom and recon images
display_slice_phantom = num_phantom_slices // 2
display_x_phantom = num_phantom_rows // 2
display_y_phantom = num_phantom_cols // 2
display_slice_recon = num_image_slices // 2
display_x_recon = num_image_rows // 2
display_y_recon = num_image_cols // 2

# sinogram images
for view_idx in [0, num_views//4, num_views//2]:
    view_angle = int(angles[view_idx]*180/np.pi)
    plot_image(sino[view_idx, :, :], title=f'sinogram view angle {view_angle} ',
               filename=os.path.join(save_path, f'sino-shepp-logan-3D-view_angle{view_angle}.png'))

# display phantom images
plot_image(phantom[display_slice_phantom], title=f'phantom, axial slice {display_slice_phantom}',
           filename=os.path.join(save_path, 'phantom_axial.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,display_x_phantom,:], title=f'phantom, coronal slice {display_x_phantom}',
           filename=os.path.join(save_path, 'phantom_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,:,display_y_phantom], title=f'phantom, sagittal slice {display_y_phantom}',
           filename=os.path.join(save_path, 'phantom_sagittal.png'), vmin=vmin, vmax=vmax)
           
# display recon images
plot_image(recon[display_slice_recon], title=f'qGGMRF recon, axial slice {display_slice_recon}',
           filename=os.path.join(save_path, 'recon_axial.png'), vmin=vmin, vmax=vmax)
plot_image(recon[:,display_x_recon,:], title=f'qGGMRF recon, coronal slice {display_x_recon}',
           filename=os.path.join(save_path, 'recon_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(recon[:,:,display_y_recon], title=f'qGGMRF recon, sagittal slice {display_y_recon}',
           filename=os.path.join(save_path, 'recon_sagittal.png'), vmin=vmin, vmax=vmax)
           
print(f"Images saved to {save_path}.") 
input("Press Enter")

