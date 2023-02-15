import os
import numpy as np
import mbircone
from demo_utils import plot_image, plot_gif


"""
This script is a demonstration of the laminography reconstruction algorithm. Demo functionality includes
 * Generating a 3D laminography sample phantom;
 * Generating synthetic laminography data by forward projecting the phantom.
 * Performing a 3D qGGMRF reconstruction.
 * Displaying the results.
"""
print('This script is a demonstration of the laminography reconstruction algorithm. Demo functionality includes \
\n\t * Generating a 3D laminography sample phantom; \
\n\t * Generating synthetic laminography data by forward projecting the phantom. \
\n\t * Performing a 3D qGGMRF reconstruction. \
\n\t * Displaying the results.')

# Laminographic angle
theta_degrees = 60
# Convert to radians
theta_radians = theta_degrees * (np.pi/180)

# detector size
num_det_channels = 64
num_det_rows = 64

# number of projection views
num_views = 128

# Phantom parameters
num_slices_phantom = 16
num_rows_phantom = 64
num_cols_phantom = 64

# Reconstruction parameters
num_slices_image = 16
num_rows_image = 146
num_cols_image = 146

# Size of constant padding around phantom as a multiple of num_rows_phantom or num_cols_phantom
pad_factor = 2

# Reconstruction size
num_image_slices = 18

# qGGMRF recon parameters
sharpness = 0.0                    # Controls regularization level of reconstruction by controlling prior term weighting
snr_db = 30

# display parameters
vmin = 0.0
vmax = 0.1


# Compute projection angles uniformly spaced within the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

# local path to save phantom, sinogram, and reconstruction images
save_path = f'output/laminography_tests/'
os.makedirs(save_path, exist_ok=True)

######################################################################################
# Generate laminography phantom, pad with 'constant' values
######################################################################################

print('Generating a laminography phantom ...')
phantom = mbircone.phantom.gen_lamino_sample_3d(num_rows_phantom, num_cols_phantom,
                                                num_slices_phantom, pad_factor=pad_factor)

# Scale the phantom by a factor of 10.0 to make the projections physical realistic -log attenuation values
phantom = phantom/10.0
print('Phantom shape is:', num_slices_phantom, num_rows_phantom, num_cols_phantom)


######################################################################################
# Generate synthetic sinogram
######################################################################################

print('Generating synthetic sinogram ...')
sino = mbircone.laminography.project_lamino(phantom, angles, theta_radians,
                                            num_det_rows, num_det_channels)
print('Synthetic sinogram shape: (num_views, num_det_rows, num_det_channels) = ', sino.shape)


######################################################################################
# Perform 3D qGGMRF reconstruction
######################################################################################

print('Performing 3D qGGMRF reconstruction ...')

recon = mbircone.laminography.recon_lamino(sino, angles, theta_radians,
                                           num_image_rows=146,
                                           num_image_cols=146,
                                           num_image_slices=16,
                                           sharpness=sharpness, snr_db=snr_db)

print('recon shape = ', np.shape(recon))


#####################################################################################
# Generate phantom, synthetic sinogram, and reconstruction images
#####################################################################################
# sinogram images
for view_idx in [0, num_views//4, num_views//2]:
        view_angle = int(angles[view_idx]*180/np.pi)
        plot_image(sino[view_idx, :, :], title=f'sinogram view angle {view_angle} ',
                   filename=os.path.join(save_path, f'sino-shepp-logan-3D-view_angle{view_angle}.png'),
                   vmin=0.0, vmax=2.0)

plot_image(sino[0,:,:]-sino[num_views//4,:,:],
           title=f'sinogram view angle {0} minus view angle'
                 f' {int(angles[num_views//4]*180/np.pi)}',
           filename=os.path.join(save_path, f'sinogram-difference.png'),
           vmin=0.0, vmax=2.0)

plot_image(np.abs(sino[0,:,:]-sino[num_views//3,:,:]),
           title=f'sinogram view angle {0} minus view angle'
                 f' {int(angles[num_views//3]*180/np.pi)}',
           filename=os.path.join(save_path, f'sinogram-difference.png'),
           vmin=0.0, vmax=2.0)

display_phantom = phantom
display_recon = recon

(num_slices_image, num_rows_image, num_cols_image) = np.shape(recon)

# Determine where the relevant phantom region begins and ends
recon_row_start = int((num_rows_image-num_rows_phantom)/2)
recon_row_end = recon_row_start + num_rows_phantom
recon_col_start = int((num_cols_image-num_cols_phantom)/2)
recon_col_end = recon_col_start + num_cols_phantom
recon_slice_start = int((num_slices_image-num_slices_phantom)/2)
recon_slice_end = recon_slice_start + num_slices_phantom
recon_area_of_interest = recon[recon_slice_start:recon_slice_end, recon_row_start:recon_row_end,
                         recon_col_start:recon_col_end]

phantom_row_start = int(num_rows_phantom * pad_factor)
phantom_row_end = phantom_row_start + num_rows_phantom
phantom_col_start = int(num_cols_phantom * pad_factor)
phantom_col_end = phantom_col_start + num_cols_phantom
phantom_area_of_interest = phantom[:, phantom_row_start:phantom_row_end, phantom_col_start:phantom_col_end]

# Compute and display reconstruction error in phantom region
display_error = np.abs(phantom_area_of_interest - recon_area_of_interest)

print(f'qGGMRF normalized rms reconstruction error within 16x128x128 laminography phantom window: '
      f'{np.sqrt(np.mean(display_error**2))/np.mean(phantom_area_of_interest):.3g}')

# Set display indexes for phantom and recon images
display_slice_image = display_phantom.shape[0] // 2
display_x_image = display_phantom.shape[1] // 2
display_y_image = display_phantom.shape[2] // 2

display_slice_recon = display_recon.shape[0] // 2
display_x_recon = display_recon.shape[1] // 2
display_y_recon = display_recon.shape[2] // 2

display_slice_error = display_error.shape[0] // 2
display_x_error = display_error.shape[1] // 2
display_y_error = display_error.shape[2] // 2

# phantom images
plot_image(display_phantom[display_slice_image], title=f'phantom, axial slice {display_slice_image}',
           filename=os.path.join(save_path, 'phantom_axial.png'), vmin=vmin, vmax=vmax)
plot_image(display_phantom[:,display_x_image,:], title=f'phantom, coronal slice {display_x_image}',
           filename=os.path.join(save_path, 'phantom_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(display_phantom[:,:,display_y_image], title=f'phantom, sagittal slice {display_y_image}',
           filename=os.path.join(save_path, 'phantom_sagittal.png'), vmin=vmin, vmax=vmax)
# recon images
plot_image(display_recon[display_slice_recon], title=f'qGGMRF recon, axial slice {display_slice_recon}, '
                                                     f'Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'recon_axial.png'), vmin=vmin, vmax=vmax)
plot_image(display_recon[:, display_x_recon,:], title=f'qGGMRF recon, coronal slice {display_x_recon}, '
                                                      f'Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'recon_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(display_recon[:, :, display_y_recon], title=f'qGGMRF recon, sagittal slice {display_y_recon}, '
                                                       f'Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'recon_sagittal.png'), vmin=vmin, vmax=vmax)

# error images
plot_image(display_error[display_slice_error], title=f'error, axial slice {display_slice_error}, '
                                                     f'Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'error_axial.png'), vmin=vmin, vmax=vmax)
plot_image(display_error[:, display_x_error,:], title=f'error, coronal slice {display_x_error}, '
                                                      f'Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'error_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(display_error[:, :, display_y_error], title=f'error, sagittal slice {display_y_error}, '
                                                       f'Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'error_sagittal.png'), vmin=vmin, vmax=vmax)


print(f"Images saved to {save_path}.")
input("Press Enter")
