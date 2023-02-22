import os
import numpy as np
import mbircone
from demo_utils import plot_image, plot_gif, create_circular_mask, nrmse


def create_circular_mask(h, w, center=None, radius=None):
    """
    Creates a circular mask for an image of size h x w, with a given center and radius.
    Adapted from code by Alexander Reynolds via StackExchange

    Args:
        h (int): Height of image in pixels
        w (int): Width of image in pixels
        center (float, 2-tuple): Coordinates of center of circular mask, in pixels.
        radius (float): Radius of circular mask, in pixels.
    Returns:
        h x w image with a circular mask centered at 'center' and with radius 'radius'
        mask = 0 if pixel is within radius, 1 if pixel is outside radius

    """
    if center is None: # use the middle of the image
        center = ((w-1)/2, (h-1)/2)
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])+(1/2)

    # Calculate distance from each pixel to the center of the circle
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    # Create mask based on cutoff for distance
    mask = dist_from_center > radius
    return mask


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
num_views = 64

# Phantom parameters
num_phantom_slices = 16
num_phantom_rows = 64
num_phantom_cols = 64

# Size of constant padding around phantom as a multiple of num_phantom_rows or num_phantom_cols
tile_rows = 5
tile_cols = 5

# Reconstruction parameters
num_image_slices = 16

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
phantom = mbircone.phantom.gen_lamino_sample_3d(num_phantom_rows, num_phantom_cols,
                                                num_phantom_slices, tile_rows=tile_rows,
                                                tile_cols=tile_cols)

# Scale the phantom by a factor of 10.0 to make the projections physical realistic -log attenuation values
phantom = phantom/10.0
print('Phantom shape is:', num_phantom_slices, num_phantom_rows, num_phantom_cols)


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
                                           num_image_slices=num_image_slices,
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

(num_image_slices, num_image_rows, num_image_cols) = np.shape(recon)

# Set display indexes for phantom and recon images
display_slice_phantom = phantom.shape[0] // 2
display_x_phantom = phantom.shape[1] // 2
display_y_phantom = phantom.shape[2] // 2

display_slice_recon = recon.shape[0] // 2
display_x_recon = recon.shape[1] // 2
display_y_recon = recon.shape[2] // 2

# phantom images
plot_image(phantom[display_slice_phantom], title=f'phantom, axial slice {display_slice_phantom}',
           filename=os.path.join(save_path, 'phantom_axial.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,display_x_phantom,:], title=f'phantom, coronal slice {display_x_phantom}',
           filename=os.path.join(save_path, 'phantom_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,:,display_y_phantom], title=f'phantom, sagittal slice {display_y_phantom}',
           filename=os.path.join(save_path, 'phantom_sagittal.png'), vmin=vmin, vmax=vmax)

# recon images
plot_image(recon[display_slice_recon], title=f'qGGMRF recon, axial slice {display_slice_recon}, '
                                                     f'Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'recon_axial.png'), vmin=vmin, vmax=vmax)
plot_image(recon[:, display_x_recon,:], title=f'qGGMRF recon, coronal slice {display_x_recon}, '
                                                      f'Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'recon_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(recon[:, :, display_y_recon], title=f'qGGMRF recon, sagittal slice {display_y_recon}, '
                                                       f'Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'recon_sagittal.png'), vmin=vmin, vmax=vmax)
plot_image(recon[0], title=f'qGGMRF recon, axial slice 0, '
                                                     f'Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'recon_axial_0.png'), vmin=vmin, vmax=vmax)

#####################################################################################
# Generate NRMSE and error images
#####################################################################################

# Determine where the relevant recon region begins and ends
recon_row_start = int((num_image_rows-num_phantom_rows)/2)
recon_row_end = recon_row_start + num_phantom_rows
recon_col_start = int((num_image_cols-num_phantom_cols)/2)
recon_col_end = recon_col_start + num_phantom_cols
recon_slice_start = int((num_image_slices-num_phantom_slices)/2)
recon_slice_end = recon_slice_start + num_phantom_slices
recon_roi = recon[recon_slice_start:recon_slice_end, recon_row_start:recon_row_end,
                recon_col_start:recon_col_end]

# Determine where the relevant phantom region begins and ends
phantom_row_start = int(num_phantom_rows * (tile_rows-1)/2)
phantom_row_end = phantom_row_start + num_phantom_rows
phantom_col_start = int(num_phantom_cols * (tile_cols-1)/2)
phantom_col_end = phantom_col_start + num_phantom_cols
phantom_roi = phantom[:, phantom_row_start:phantom_row_end, phantom_col_start:phantom_col_end]

# Compute and display reconstruction error in cylindrical phantom region
mask = create_circular_mask(num_phantom_rows, num_phantom_cols)
mask = np.tile(mask, (num_phantom_slices, 1, 1))
masked_recon_roi = np.ma.masked_where(mask, recon_roi)
masked_phantom_roi = np.ma.masked_where(mask, phantom_roi)

nrmse = nrmse(masked_recon_roi, masked_phantom_roi)
print(f'qGGMRF normalized rms reconstruction error within laminography phantom window of diameter 64: '
      f'{nrmse:.3g}')

# Generate image representing error
error = np.abs(masked_recon_roi - masked_phantom_roi)

display_slice_error = error.shape[0] // 2
display_x_error = error.shape[1] // 2
display_y_error = error.shape[2] // 2

# error images
plot_image(error[display_slice_error], title=f'error, axial slice {display_slice_error}, '
                                                     f'Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'error_axial.png'), vmin=0.0, vmax=0.05, cmap='viridis')
plot_image(error[:, display_x_error,:], title=f'error, coronal slice {display_x_error}, '
                                                      f'Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'error_coronal.png'), vmin=0.0, vmax=0.05, cmap='viridis')
plot_image(error[:, :, display_y_error], title=f'error, sagittal slice {display_y_error}, '
                                                       f'Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'error_sagittal.png'), vmin=0.0, vmax=0.05, cmap='viridis')

print(f"Images saved to {save_path}.")
input("Press Enter")
