import os
import numpy as np
import mbircone
from demo_utils import plot_image, plot_gif

# laminographic angle
theta_degrees = 60
theta = theta_degrees * (np.pi/180) # Convert to radians

# detector size
num_det_channels = 128
num_det_rows = 128

# number of projection views
num_views = 128

# Phantom parameters
num_slices_phantom = 16
num_rows_phantom = 90
num_cols_phantom = 90

# Reconstruction size
#num_image_rows=int( num_det_channels + 2 * num_slices_phantom * sin(theta) )
num_image_rows=128
num_image_cols=128
num_image_slices=32

# qGGMRF recon parameters
sharpness = 0.0                     # Controls regularization level of reconstruction by controling prior term weighting
snr_db = 30

# display parameters
vmin = 0.0
vmax = 0.1


# Compute projection angles uniformly spaced within the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

# local path to save phantom, sinogram, and reconstruction images
save_path = f'output/laminography_tests/'
os.makedirs(save_path, exist_ok=True)

print('Generating a laminography phantom ...')
phantom = mbircone.phantom.gen_lamino_sample_3d(num_rows_phantom, num_cols_phantom, num_slices_phantom)

# Scale the phantom by a factor of 10.0 to make the projections physical realistic -log attenuation values
phantom = phantom/10.0
print('Phantom shape is:', num_slices_phantom, num_rows_phantom, num_cols_phantom)

# Pad phantom so that the edges are replicated
pad_factor = 2
pad_rows_L = num_rows_phantom * pad_factor
pad_rows_R = pad_rows_L
pad_cols_L = num_cols_phantom * pad_factor
pad_cols_R = pad_cols_L
pad_slices_L = num_slices_phantom * pad_factor
pad_slices_R = pad_slices_L

phantom = np.pad(phantom, [(0, 0), (pad_rows_L, pad_rows_R),(pad_cols_L, pad_cols_R)], mode='edge')
phantom = np.pad(phantom, [(pad_slices_L, pad_slices_R), (0, 0), (0, 0)], mode='constant', constant_values=0.0)


######################################################################################
# Generate synthetic sinogram
######################################################################################
print('Generating synthetic sinogram ...')
sino = mbircone.cone3D.project(phantom, angles, num_det_rows, num_det_channels, None, 1,
                               geometry='lamino', theta=theta)
print('Synthetic sinogram shape: (num_views, num_det_rows, num_det_channels) = ', sino.shape)


######################################################################################
# Perform 3D qGGMRF reconstruction
######################################################################################
print('Performing 3D qGGMRF reconstruction ...')
recon = mbircone.cone3D.recon(sino, angles, None, None, geometry='lamino', theta=theta, sharpness=sharpness, snr_db=snr_db,
                              num_image_rows=num_image_rows, num_image_cols=num_image_cols, num_image_slices=num_image_slices)
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

display_phantom = phantom
display_recon = recon

# Set display indexes for phantom and recon images
display_slice_image = display_phantom.shape[0] // 2
display_x_image = display_phantom.shape[1] // 2
display_y_image = display_phantom.shape[2] // 2

display_slice_recon = display_recon.shape[0] // 2
display_x_recon = display_recon.shape[1] // 2
display_y_recon = display_recon.shape[2] // 2

# phantom images
plot_image(display_phantom[display_slice_image], title=f'phantom, axial slice {display_slice_image}',
           filename=os.path.join(save_path, 'phantom_axial.png'), vmin=vmin, vmax=vmax)
plot_image(display_phantom[display_slice_image], title=f'phantom, axial slice {display_slice_image}',
           filename=os.path.join(save_path, 'phantom_axial.png'), vmin=vmin, vmax=vmax)
plot_image(display_phantom[:,display_x_image,:], title=f'phantom, coronal slice {display_x_image}',
           filename=os.path.join(save_path, 'phantom_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(display_phantom[:,:,display_y_image], title=f'phantom, sagittal slice {display_y_image}',
           filename=os.path.join(save_path, 'phantom_sagittal.png'), vmin=vmin, vmax=vmax)
# recon images
plot_image(display_recon[display_slice_recon], title=f'qGGMRF recon, axial slice {display_slice_recon}, Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'recon_axial.png'), vmin=0, vmax=0.40)
plot_image(display_recon[display_slice_recon], title=f'qGGMRF recon, axial slice {display_slice_recon}, Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'recon_axial.png'), vmin=vmin, vmax=vmax)
plot_image(display_recon[:, display_x_recon,:], title=f'qGGMRF recon, coronal slice {display_x_recon}, Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'recon_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(display_recon[:, :, display_y_recon], title=f'qGGMRF recon, sagittal slice {display_y_recon}, Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'recon_sagittal.png'), vmin=vmin, vmax=vmax)

print(f"Images saved to {save_path}.")
input("Press Enter")
