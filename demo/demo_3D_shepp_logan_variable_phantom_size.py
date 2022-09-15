import os
import numpy as np
import mbircone
from demo_utils import plot_image, plot_gif, plt_cmp_3dobj

"""
This script is a demonstration of the 3D qGGMRF reconstruction algorithm. Demo functionality includes
 * generating a 3D Shepp Logan phantom, generating synthetic sinogram data by
 * forward projecting the phantom, performing two 3D qGGMRF reconstructions with variable ROR sizes,
 * and displaying the results.
"""
print('This script is a demonstration of the 3D qGGMRF reconstruction algorithm. Demo functionality includes \
\n\t * generating a 3D Shepp Logan phantom, generating synthetic sinogram data by \
\n\t * forward projecting the phantom, performing two 3D qGGMRF reconstructions with variable ROR sizes, \
\n\t * and displaying the results.')

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

# Size of phantom A
num_slices_phantom_A = 128
num_rows_phantom_A = 128
num_cols_phantom_A = 128

# Size of phantom B
num_slices_phantom_B = 192
num_rows_phantom_B = 192
num_cols_phantom_B = 192

delta_pixel_phantom = 1.0

# Size of recon A
num_slices_recon_A = 128
num_rows_recon_A = 128
num_cols_recon_A = 128

# Size of recon B
num_slices_recon_B = 192
num_rows_recon_B = 192
num_cols_recon_B = 192

delta_pixel_recon = 1.0

# local path to save phantom, sinogram, and reconstruction images
save_path = f'output/3D_shepp_logan_variable_phantom_size/'
os.makedirs(save_path, exist_ok=True)

print('Genrating 3D Shepp Logan phantom ...')
######################################################################################
# Generate a 3D shepp logan phantom
######################################################################################

phantom_A = mbircone.phantom.gen_shepp_logan_3d(num_rows_phantom_A, num_cols_phantom_A, num_slices_phantom_A)
# scale the phantom by a factor of 10.0 to make the projections physical realistic -log attenuation values
phantom_A = phantom_A/10.0
print('Phantom A shape = ', np.shape(phantom_A))

phantom_B = mbircone.phantom.gen_shepp_logan_3d(num_rows_phantom_B, num_cols_phantom_B, num_slices_phantom_B)
# scale the phantom by a factor of 10.0 to make the projections physical realistic -log attenuation values
phantom_B = phantom_B/10.0
print('Phantom B shape = ', np.shape(phantom_B))

######################################################################################
# Generate synthetic sinogram
######################################################################################

print('Generating synthetic sinograms ...')
sino_A = mbircone.cone3D.project(phantom_A, angles,
                               num_det_rows, num_det_channels, 
                               dist_source_detector, magnification, delta_pixel_image=delta_pixel_phantom)
print('Synthetic sinogram shape, phantom A: (num_views, num_det_rows, num_det_channels) = ', sino_A.shape)

sino_B = mbircone.cone3D.project(phantom_B, angles,
                               num_det_rows, num_det_channels,
                               dist_source_detector, magnification, delta_pixel_image=delta_pixel_phantom)
print('Synthetic sinogram shape, phantom B: (num_views, num_det_rows, num_det_channels) = ', sino_B.shape)

######################################################################################
# Perform 3D qGGMRF reconstruction
######################################################################################

print('Performing 3D qGGMRF reconstruction ...')

recon_A = mbircone.cone3D.recon(sino_A, angles, dist_source_detector, magnification, delta_pixel_image=delta_pixel_recon, num_rows=num_rows_recon_A, num_cols=num_cols_recon_A, num_slices=num_slices_recon_A, sharpness=sharpness, T=T)

print('recon A shape = ', np.shape(recon_A))

recon_B = mbircone.cone3D.recon(sino_B, angles, dist_source_detector, magnification, delta_pixel_image=delta_pixel_recon, num_rows=num_rows_recon_B, num_cols=num_cols_recon_B, num_slices=num_slices_recon_B, sharpness=sharpness, T=T)

print('recon B shape = ', np.shape(recon_B))

######################################################################################
# Generate phantom, synthetic sinogram, and reconstruction images
######################################################################################
# sinogram images
for view_idx in [0, num_views//4, num_views//2]:
    view_angle = int(angles[view_idx]*180/np.pi)
    plot_image(sino_A[view_idx, :, :], title=f'sinogram A view angle {view_angle} ',
               filename=os.path.join(save_path, f'sino-shepp-logan-3D-view_angle{view_angle}_sinogram_A.png'))
for view_idx in [0, num_views//4, num_views//2]:
    view_angle = int(angles[view_idx]*180/np.pi)
    plot_image(sino_B[view_idx, :, :], title=f'sinogram B view angle {view_angle} ',
               filename=os.path.join(save_path, f'sino-shepp-logan-3D-view_angle{view_angle}_sinogram_B.png'))
# Set display indexes for phantom and recon images
display_slice_phantom_A = num_slices_phantom_A // 2
display_x_phantom_A = num_rows_phantom_A // 2
display_y_phantom_A = num_cols_phantom_A // 2
display_slice_phantom_B = num_slices_phantom_B // 2
display_x_phantom_B = num_rows_phantom_B // 2
display_y_phantom_B = num_cols_phantom_B // 2
display_slice_recon_A = num_slices_recon_A // 2
display_x_recon_A = num_rows_recon_A // 2
display_y_recon_A = num_cols_recon_A // 2
display_slice_recon_B = num_slices_recon_B // 2
display_x_recon_B = num_rows_recon_B // 2
display_y_recon_B = num_cols_recon_B // 2

# phantom images
plot_image(phantom_A[display_slice_phantom_A], title=f'phantom A, axial slice {display_slice_phantom_A}',
           filename=os.path.join(save_path, 'phantom_A_axial.png'), vmin=vmin, vmax=vmax)
plot_image(phantom_A[:,display_x_phantom_A,:], title=f'phantom A, coronal slice {display_x_phantom_A}',
           filename=os.path.join(save_path, 'phantom_A_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(phantom_A[:,:,display_y_phantom_A], title=f'phantom A, sagittal slice {display_y_phantom_A}',
           filename=os.path.join(save_path, 'phantom_A_sagittal.png'), vmin=vmin, vmax=vmax)
           
# recon images
plot_image(recon_A[display_slice_recon_A], title=f'qGGMRF recon A, axial slice {display_slice_recon_A}',
           filename=os.path.join(save_path, 'recon_A_axial.png'), vmin=vmin, vmax=vmax)
plot_image(recon_A[:,display_x_recon_A,:], title=f'qGGMRF recon A, coronal slice {display_x_recon_A}',
           filename=os.path.join(save_path, 'recon_A_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(recon_A[:,:,display_y_recon_A], title=f'qGGMRF recon A, sagittal slice {display_y_recon_A}',
           filename=os.path.join(save_path, 'recon_A_sagittal.png'), vmin=vmin, vmax=vmax)
         
# phantom images
plot_image(phantom_B[display_slice_phantom_B], title=f'phantom B, axial slice {display_slice_phantom_B}',
           filename=os.path.join(save_path, 'phantom_B_axial.png'), vmin=vmin, vmax=vmax)
plot_image(phantom_B[:,display_x_phantom_B,:], title=f'phantom B, coronal slice {display_x_phantom_B}',
           filename=os.path.join(save_path, 'phantom_B_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(phantom_B[:,:,display_y_phantom_B], title=f'phantom B, sagittal slice {display_y_phantom_B}',
           filename=os.path.join(save_path, 'phantom_B_sagittal.png'), vmin=vmin, vmax=vmax)
         
         
# recon images
plot_image(recon_B[display_slice_recon_B], title=f'qGGMRF recon B, axial slice {display_slice_recon_B}',
          filename=os.path.join(save_path, 'recon_B_axial.png'), vmin=vmin, vmax=vmax)
plot_image(recon_B[:,display_x_recon_B,:], title=f'qGGMRF recon B, coronal slice {display_x_recon_B}',
          filename=os.path.join(save_path, 'recon_B_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(recon_B[:,:,display_y_recon_B], title=f'qGGMRF recon B, sagittal slice {display_y_recon_B}',
          filename=os.path.join(save_path, 'recon_B_sagittal.png'), vmin=vmin, vmax=vmax)
           
print(f"Images saved to {save_path}.") 
input("Press Enter")

