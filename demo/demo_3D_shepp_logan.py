import os
import numpy as np
import mbircone
from demo_utils import plot_image, plot_gif, plt_cmp_3dobj

# Set sinogram shape
num_det_rows = 128
num_det_channels = 128
num_views = 144

# Reconstruction parameters
sharpness = 1.0                                               # Controls sharpness of reconstruction
magnification = 2.0                                           # Ratio of (source to detector)/(source to center of rotation)
dist_source_detector = 2*num_det_channels                     # distance from source to detector in ALU
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False) # set of projection angles

# Display parameters
vmin = 1.0  # minimum display value
vmax = 1.1  # maximum display value
filename = 'output/3D_shepp_logan/results.png'

# Generate a 3D shepp logan phantom.
ROR, boundary_size = mbircone.cone3D.compute_img_size(num_views, num_det_rows, num_det_channels, dist_source_detector, magnification)
Nz, Nx, Ny = ROR   # function ensures that Nx=Ny
img_slices_boundary_size, img_rows_boundary_size, img_cols_boundary_size = boundary_size
print('ROR of the recon is:', (Nz, Nx, Ny))

# Determine required phantom size based on ROI.
# ROI is determine by ROR - 2*boundary_size
num_rows_cols = Nx - 2 * img_rows_boundary_size                                         # determines the width and height of the ROI 
num_slices_phantom = Nz - 2 * img_slices_boundary_size                                  # determines the depth of the ROI
print('ROI and shape of phantom is:', num_slices_phantom, num_rows_cols, num_rows_cols) # assumes that width=height

# Set display indexes
display_slice = img_slices_boundary_size + int(0.4*num_slices_phantom)
display_x = num_rows_cols // 2
display_y = num_rows_cols // 2
display_view = 0

# Generate a 3D shepp logan phantom
phantom = mbircone.phantom.gen_shepp_logan_3d(num_rows_cols, num_rows_cols, num_slices_phantom) # generate phatom within ROI
print('Generated ROI phantom shape = ', np.shape(phantom))
phantom = mbircone.cone3D.pad_roi2ror(phantom, boundary_size)                                   # generate phantom wihtin ROR
print('Padded ROR phantom shape = ', np.shape(phantom))

# Compute the forward projection of the phantom
sino = mbircone.cone3D.project(phantom, angles, num_det_rows, num_det_channels, dist_source_detector, magnification)

# Reconstruction sinogram data
recon = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification, sharpness=sharpness )

# create output folder
os.makedirs('output/3D_shepp_logan/', exist_ok=True)

print('sino shape = ', np.shape(sino), sino.dtype)
plot_image(sino[display_view, :, :], title='sino', filename='output/3D_shepp_logan/sino-shepp-logan-3D-view(%.2f).png' % angles[0])
# plot_gif(sino, 'output', 'sino-shepp-logan-3D')

print('recon shape = ', np.shape(recon))
np.save('output/3D_shepp_logan/recon.npy', recon)

#Display and compare reconstruction
plt_cmp_3dobj(phantom, recon, display_slice, display_x, display_y, vmin, vmax, filename)

input("press Enter")
