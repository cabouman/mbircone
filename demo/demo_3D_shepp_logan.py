import os
import numpy as np
import mbircone
from demo_utils import plot_image, plot_gif, plt_cmp_3dobj

"""
This script is a demonstration of the 3D qGGMRF reconstruction algorithm. Demo functionality includes
 * generating a 3D Shepp Logan phantom, and zero-padding it to the size of ROR
 * generating synthetic sinogram data by forward projecting the phantom
 * performing a 3D qGGMRF reconstructions and displaying the results.
"""
print('This script is a demonstration of the 3D qGGMRF reconstruction algorithm. Demo functionality includes \
\n\t * generating a 3D Shepp Logan phantom, and zero-padding it to the size of ROR \
\n\t * generating synthetic sinogram data by forward projecting the phantom \
\n\t * performing a 3D qGGMRF reconstructions and displaying the results.')

# ###########################################################################
# Set the parameters to generate the phantom, synthetic sinogram, and do the recon
# ###########################################################################

# Change the parameters below for your own use case.

# Detector size
num_det_rows = 128
num_det_channels = 128
# Geometry parameters
magnification = 1.0                         # Ratio of (source to detector)/(source to center of rotation)
dist_source_detector = 10*num_det_channels  # distance from source to detector in ALU
# number of projection views
num_views = 64
# projection angles will be uniformly spaced within the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
# qGGMRF recon parameters
sharpness = 0.0                             # Controls regularization: larger => sharper; smaller => smoother
T = 0.1                                     # Controls edginess of reconstruction
# display parameters
vmin = 0.10
vmax = 0.12

# Size of phantom
num_slices_phantom = 128
num_rows_phantom = 128
num_cols_phantom = 128
delta_pixel_phantom = 1.0

# Size of recon
num_slices_recon = 128
num_rows_recon = 128
num_cols_recon = 128
delta_pixel_recon = 1.0

# local path to save phantom, sinogram, and reconstruction images
save_path = f'output/3D_shepp_logan/'
os.makedirs(save_path, exist_ok=True)

print('Genrating 3D Shepp Logan phantom ...')

######################################################################################
# Generate a 3D shepp logan phantom
######################################################################################
phantom = mbircone.phantom.gen_shepp_logan_3d(num_rows_phantom, num_cols_phantom, num_slices_phantom)
# scale the phantom by a factor of 10.0 to make the projections physical realistic -log attenuation values
phantom = phantom/10.0
print('Phantom shape = ', np.shape(phantom))

######################################################################################
# Generate synthetic sinogram
######################################################################################
print('Generating synthetic sinogram ...')
sino = mbircone.cone3D.project(phantom, angles, 
                               num_det_rows, num_det_channels, 
                               dist_source_detector, magnification, delta_pixel_image=delta_pixel_phantom)
print('Synthetic sinogram shape: (num_views, num_det_rows, num_det_channels) = ', sino.shape)

######################################################################################
# Perform 3D qGGMRF reconstruction
######################################################################################

print('Performing 3D qGGMRF reconstruction ...')

recon = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification, delta_pixel_image=delta_pixel_recon, num_rows=num_rows_recon, num_cols=num_cols_recon, num_slices=num_slices_recon, sharpness=sharpness, T=T)

print('recon shape = ', np.shape(recon))

######################################################################################
# Generate phantom, synthetic sinogram, and reconstruction images
######################################################################################
# sinogram images
for view_idx in [0, num_views//4, num_views//2]:
    view_angle = int(angles[view_idx]*180/np.pi)
    plot_image(sino[view_idx, :, :], title=f'sinogram view angle {view_angle} ', 
               filename=os.path.join(save_path, f'sino-shepp-logan-3D-view_angle{view_angle}.png'))
# Set display indexes for phantom and recon images
display_slice_phantom = num_slices_phantom // 2
display_x_phantom = num_rows_phantom // 2
display_y_phantom = num_cols_phantom // 2
display_slice_recon = num_slices_recon // 2
display_x_recon = num_rows_recon // 2
display_y_recon = num_cols_recon // 2
# phantom images
plot_image(phantom[display_slice_phantom], title=f'phantom, axial slice {display_slice_phantom}',
           filename=os.path.join(save_path, 'phantom_axial.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,display_x_phantom,:], title=f'phantom, coronal slice {display_x_phantom}',
           filename=os.path.join(save_path, 'phantom_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,:,display_y_phantom], title=f'phantom, sagittal slice {display_y_phantom}',
           filename=os.path.join(save_path, 'phantom_sagittal.png'), vmin=vmin, vmax=vmax)
# recon images
plot_image(recon[display_slice_recon], title=f'qGGMRF recon, axial slice {display_slice_recon}',
           filename=os.path.join(save_path, 'recon_axial.png'), vmin=vmin, vmax=vmax)
plot_image(recon[:,display_x_recon,:], title=f'qGGMRF recon, coronal slice {display_x_recon}',
           filename=os.path.join(save_path, 'recon_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(recon[:,:,display_y_recon], title=f'qGGMRF recon, sagittal slice {display_y_recon}',
           filename=os.path.join(save_path, 'recon_sagittal.png'), vmin=vmin, vmax=vmax)
print(f"Images saved to {save_path}.") 
input("Press Enter")

