import os
import numpy as np
import mbircone
from demo_utils import plot_image, plot_gif, plt_cmp_3dobj


# ###########################################################################
# Set the parameters to generate the phantom, synthetic sinogram, and do the recon
# ###########################################################################

# Change the parameters below for your own use case.

# Laminographic angle in radians. Must be at least 25 degrees here.

theta_degrees = 60

theta = theta_degrees * (np.pi/180)

# detector size
num_det_rows = 64
num_det_channels = 64
# Geometry parameters


# number of projection views
num_views = 128
# projection angles will be uniformly spaced within the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
# qGGMRF recon parameters
sharpness = 1.0                             # Controls regularization level of reconstruction by controling prior term weighting
T = 0.1                                     # Controls edginess of reconstruction
# convergence parameters
stop_threshold = 0.005
# display parameters
vmin = 0.10
vmax = 0.12

# local path to save phantom, sinogram, and reconstruction images
save_path = f'output/3D_shepp_logan_lamino/'
os.makedirs(save_path, exist_ok=True)


print('Genrating 3D Shepp Logan phantom ...')
######################################################################################
# This section determines the phantom size corresponding to the geometry parameters
######################################################################################
# Compute ROR (Region of Reconstruction) and boundary size
(num_slices_ROR, num_rows_ROR, num_cols_ROR), boundary_size = mbircone.lamino.compute_img_size_lamino(num_views, num_det_rows, num_det_channels, theta)

# Compute ROI (Region of Interest) size from ROR and boundary size
# In principle the object of interest should be within ROI.

#num_slices_ROI = int(num_det_rows * 0.22)
#num_slices_ROI += (num_slices_ROR-num_slices_ROI)%2

num_slices_ROI = num_slices_ROR - 2*boundary_size[0]
num_rows_ROI = num_rows_ROR - 2*boundary_size[1]
num_cols_ROI = num_cols_ROR - 2*boundary_size[2]
#
#boundary_slices = boundary_size[0]
#boundary_size[0] = (num_slices_ROR - num_slices_ROI)//2
#


print('ROI shape is:', num_slices_ROI, num_rows_ROI, num_cols_ROI)


######################################################################################
# Generate a 3D shepp logan phantom with size calculated in previous section
######################################################################################
phantom = mbircone.phantom.gen_shepp_logan_3d(num_rows_ROI, num_cols_ROI, num_slices_ROI)       # generate phantom within ROI

phantom = mbircone.cone3D.pad_roi2ror(phantom, boundary_size)
# zero-pad phantom to ROR

# scale the phantom by a factor of 10.0 to make the projections physical realistic -log attenuation values
phantom = phantom/10.0
print('Padded ROR phantom shape = ', np.shape(phantom))


######################################################################################
# Generate synthetic sinogram
######################################################################################
print('Generating synthetic sinogram ...')
sino = mbircone.lamino.project_lamino(phantom, angles, theta,
                                                             num_det_rows, num_det_channels)
print('Synthetic sinogram shape: (num_views, num_det_rows, num_det_channels) = ', sino.shape)


######################################################################################
# Perform 3D qGGMRF reconstruction
######################################################################################
print('Performing 3D qGGMRF reconstruction ...')
recon = mbircone.lamino.recon_lamino(sino, angles, theta, sharpness=sharpness, T = T, stop_threshold = stop_threshold)
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
display_slice = recon.shape[0] // 2
display_x = recon.shape[1] // 2
display_y = recon.shape[2] // 2
# phantom images
plot_image(phantom[display_slice], title=f'phantom, axial slice {display_slice}',
                     filename=os.path.join(save_path, 'phantom_axial.png'), vmin=0, vmax=0.40)
plot_image(phantom[display_slice], title=f'phantom, axial slice {display_slice}',
                     filename=os.path.join(save_path, 'phantom_axial.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,display_x,:], title=f'phantom, coronal slice {display_x}',
                     filename=os.path.join(save_path, 'phantom_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,:,display_y], title=f'phantom, sagittal slice {display_y}',
                     filename=os.path.join(save_path, 'phantom_sagittal.png'), vmin=vmin, vmax=vmax)
# recon images
plot_image(recon[display_slice], title=f'qGGMRF recon, axial slice {display_slice}, Θ='+str(theta_degrees)+' degrees',
                     filename=os.path.join(save_path, 'recon_axial.png'), vmin=0, vmax=0.40)
plot_image(recon[display_slice], title=f'qGGMRF recon, axial slice {display_slice}, Θ='+str(theta_degrees)+' degrees',
                     filename=os.path.join(save_path, 'recon_axial.png'), vmin=vmin, vmax=vmax)
plot_image(recon[:,display_x,:], title=f'qGGMRF recon, coronal slice {display_x}, Θ='+str(theta_degrees)+' degrees',
                     filename=os.path.join(save_path, 'recon_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(recon[:,:,display_y], title=f'qGGMRF recon, sagittal slice {display_y}, Θ='+str(theta_degrees)+' degrees',
                     filename=os.path.join(save_path, 'recon_sagittal.png'), vmin=vmin, vmax=vmax)
print(f"Images saved to {save_path}.")
input("Press Enter")
