import os
import numpy as np
import mbircone
from demo_utils import plot_image, plot_gif, plt_cmp_3dobj, nrmse

######################################################################################
# Set up simulated scan parameters.
######################################################################################

# Set synthetic sinogram and geometry parameters
num_det_rows = 128
num_det_channels = 128
num_views = 64
dist_source_detector = 10*num_det_channels                     # distance from source to detector in ALU
magnification = 2.0                                           # Ratio of (source to detector)/(source to center of rotation)
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)  # projection angles

# Set reconstruction parameters
sharpness = 1.0                                               # Controls sharpness of reconstruction


######################################################################################
# This section determines the required phantom size based on the parameters set above.
######################################################################################
# Compute ROR (Region of Reconstruction) and boundary size
ROR, boundary_size = mbircone.cone3D.compute_img_size(num_views, num_det_rows, num_det_channels, dist_source_detector, magnification)
Nz, Nx, Ny = ROR   # function ensures that Nx=Ny
img_slices_boundary_size, img_rows_boundary_size, img_cols_boundary_size = boundary_size
print('ROR of the recon is:', (Nz, Nx, Ny))

# Compute ROI (Region of Interest) from ROR and boundary size
num_rows_cols = Nx - 2 * img_rows_boundary_size                                         # determines the width and height of the ROI 
num_slices_phantom = Nz - 2 * img_slices_boundary_size                                  # determines the depth of the ROI
print('ROI and shape of phantom is:', num_slices_phantom, num_rows_cols, num_rows_cols) # assumes that width=height

######################################################################################
# Generate a 3D shepp logan phantom and its sinogram data
######################################################################################
phantom = mbircone.phantom.gen_shepp_logan_3d(num_rows_cols, num_rows_cols, num_slices_phantom) # generate phatom within ROI
print('Generated ROI phantom shape = ', np.shape(phantom))
phantom = mbircone.cone3D.pad_roi2ror(phantom, boundary_size)                                   # generate phantom wihtin ROR
print('Padded ROR phantom shape = ', np.shape(phantom))

# Compute the forward projection of the phantom
sino = mbircone.cone3D.project(phantom, angles, num_det_rows, num_det_channels, dist_source_detector, magnification)


######################################################################################
# Generate a 3D shepp logan phantom and its sinogram data
# Perform qGGMRF reconstruction
######################################################################################
recon = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification, sharpness=sharpness, max_iterations=20, stop_threshold=0.0, init_image=phantom)

Ax = mbircone.cone3D.project(recon, angles, num_det_rows, num_det_channels, dist_source_detector, magnification)
nrmse_sino = nrmse(Ax, sino)
print(f"forward projection NRMSE = {nrmse_sino:.5f}")
sino_err = sino-Ax

######################################################################################
# Display and compare sinogram and reconstruction results.
######################################################################################
# path to save sino and recon images
save_path = f'output/3D_shepp_logan_sharpness{sharpness}_perfect_init/'
os.makedirs(save_path, exist_ok=True)
# Set display indexes
display_slice = img_slices_boundary_size + int(0.4*num_slices_phantom)
display_x = num_rows_cols // 2
display_y = num_rows_cols // 2
display_view_list = [0, num_views//4, num_views//2]
# generate sinogram plots
for display_view in display_view_list:
    plot_image(sino[display_view, :, :], title=f'sino, view angle {angles[display_view]:.1f}', filename=os.path.join(save_path, f'sino-view_angle{angles[display_view]:.1f}.png'))
    plot_image(sino_err[display_view, :, :], title=f'forward projection error, view angle {angles[display_view]:.1f}, NRMSE={nrmse_sino:.5f}', filename=os.path.join(save_path, f'forward_pj_error_view_angle{angles[display_view]:.1f}.png'), vmin=-np.max(sino_err), vmax=np.max(sino_err))
# Set display parameters for recon images
vmin = 1.0  # minimum display value
vmax = 1.1  # maximum display value
# Display recon results
plt_cmp_3dobj(phantom, recon, display_slice, display_x, display_y, vmin, vmax, filename=os.path.join(save_path, 'results.png'))

input("press Enter")
