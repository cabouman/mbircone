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
# User selectable sinogram parameters
# ###########################################################################
# Detector size
num_det_rows = 128
num_det_channels = 128
# Scanner geometry parameters
magnification = 2.0                         # Ratio of (source to detector)/(source to center of rotation)
dist_source_detector = 10*num_det_channels  # distance from source to detector in ALU
# Number of projection views
num_views = 64

# ###########################################################################
# User selectable reconstruction parameters
# ###########################################################################
# This code allows the reconstruction and phantom to have different resolutions
recon_pixel_pitch_factor = 1.0              # Ratio of pixel pitch to detector resolution at iso
delta_pixel_recon = recon_pixel_pitch_factor*(1.0/magnification) # pixel pitch of reconstruction
# qGGMRF recon parameters
sharpness = 0.0                             # Controls regularization: larger => sharper; smaller => smoother
T = 0.1                                     # Controls edginess of reconstruction
# display parameters
vmin = 0.10
vmax = 0.12

# Generate view angles uniformly in the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

# local path to save phantom, sinogram, and reconstruction images
save_path = f'output/3D_shepp_logan_customized_recon_pixel_pitch_recon/'
os.makedirs(save_path, exist_ok=True)


print('Genrating 3D Shepp Logan phantom ...')
######################################################################################
# Determine phantom size assuming default pixel pitch 
######################################################################################
# Compute ROR (Region of Reconstruction) and boundary size
(num_slices_ROR, num_rows_ROR, num_cols_ROR), boundary_size = mbircone.cone3D.compute_img_size(num_views, num_det_rows, num_det_channels, dist_source_detector, magnification)

# Compute ROI (Region of Interest) size from ROR and boundary size
# In principle the object of interest should be within ROI.
num_slices_ROI = num_slices_ROR - 2*boundary_size[0]             
num_rows_ROI = num_rows_ROR - 2*boundary_size[1]
num_cols_ROI = num_cols_ROR - 2*boundary_size[2]
print('Phantom ROI shape is: (num_slices_ROI, num_rows_ROI, num_cols_ROI) = ', num_slices_ROI, num_rows_ROI, num_cols_ROI)


######################################################################################
# Generate a 3D shepp logan phantom with size calculated in previous section
######################################################################################
phantom = mbircone.phantom.gen_shepp_logan_3d(num_rows_ROI, num_cols_ROI, num_slices_ROI)       # generate phantom within ROI
phantom = mbircone.cone3D.pad_roi2ror(phantom, boundary_size)                                   # zero-pad phantom to ROR
# Scale Shepp-Logan phantom so that it generates physically realistic -log attenuation values
phantom = phantom/10
print('Padded phantom ROR shape: (num_slices_ROR, num_rows_ROR, num_cols_ROR) = ', np.shape(phantom))


######################################################################################
# Generate synthetic sinogram
######################################################################################
print('Generating synthetic sinogram ...')
sino = mbircone.cone3D.project(phantom, angles, 
                               num_det_rows, num_det_channels, 
                               dist_source_detector, magnification)
print('Synthetic sinogram shape: (num_views, num_det_rows, num_det_channels) = ', sino.shape)


######################################################################################
# Perform 3D qGGMRF reconstruction
######################################################################################
print('Performing 3D qGGMRF reconstruction ...')
recon = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification, 
                              delta_pixel_image=delta_pixel_recon,
                              sharpness=sharpness, T=T)
print('recon shape: (num_slices, num_rows, num_cols) = ', np.shape(recon))


######################################################################################
# Display, phantom, sinogram, and reconstructed images
######################################################################################

# Display sinogram images
for view_idx in [0, num_views//4, num_views//2]:
    view_angle = int(angles[view_idx]*180/np.pi)
    plot_image(sino[view_idx, :, :], title=f'sinogram view angle {view_angle} ', 
               filename=os.path.join(save_path, f'sino-shepp-logan-3D-view_angle{view_angle}.png'))

# Choose the approximate middle slice of the phantom for display purposes
display_slice = num_slices_ROR // 2
display_x = num_rows_ROR // 2
display_y = num_cols_ROR // 2

# Display phantom images
plot_image(phantom[display_slice], title=f'phantom, axial slice {display_slice}',
           filename=os.path.join(save_path, 'phantom_axial.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,display_x,:], title=f'phantom, coronal slice {display_x}',
           filename=os.path.join(save_path, 'phantom_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,:,display_y], title=f'phantom, sagittal slice {display_y}', 
           filename=os.path.join(save_path, 'phantom_sagittal.png'), vmin=vmin, vmax=vmax)

# Choose the approximate middle slice of the recon for display purposes
display_slice = np.shape(recon)[0] //2
display_x = np.shape(recon)[1] //2
display_y = np.shape(recon)[2] //2

# Display recon images
plot_image(recon[display_slice], title=f'qGGMRF recon, axial slice {display_slice}',
           filename=os.path.join(save_path, 'recon_axial.png'), vmin=vmin, vmax=vmax)
plot_image(recon[:,display_x,:], title=f'qGGMRF recon, coronal slice {display_x}', 
           filename=os.path.join(save_path, 'recon_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(recon[:,:,display_y], title=f'qGGMRF recon, sagittal slice {display_y}', 
           filename=os.path.join(save_path, 'recon_sagittal.png'), vmin=vmin, vmax=vmax)
print(f"Images saved to {save_path}.") 
input("Press Enter")
