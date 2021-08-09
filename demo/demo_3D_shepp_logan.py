import os
import numpy as np
import mbircone
from demo_utils import plot_image, plot_gif

# Set sinogram shape
num_det_rows = 400
num_det_channels = 256
num_views = 288

# Reconstruction parameters
sharpness = 0
snr_db = 30.0

# magnification is unitless.
magnification = 2

# All distances are in unit of 1 ALU = 1 mm.
dist_source_detector = 600
delta_pixel_detector = 0.45
delta_pixel_image = 0.5
channel_offset = 0
row_offset = 0

# Display parameters
vmin = 1.0
vmax = 1.2


# Generate a 3D shepp logan phantom.
ROR, boundary_size= mbircone.cone3D.compute_img_size(num_views, num_det_rows, num_det_channels,
                                              dist_source_detector,
                                              magnification,
                                              channel_offset=channel_offset, row_offset=row_offset,
                                              delta_pixel_detector=delta_pixel_detector,
                                              delta_pixel_image=delta_pixel_image)
Nz, Nx, Ny = ROR
img_slices_boundary_size, img_rows_boundary_size, img_cols_boundary_size = boundary_size
print('ROR of the recon is:', (Nz, Nx, Ny))

# Set phantom Shape equal to ROI according to ROR and boundary_size.
# All valid pixel should inside ROI.
num_rows_cols = Nx - 2 * img_rows_boundary_size  # Assumes a square image
num_slices_phantom = Nz - 2 * img_slices_boundary_size
print('ROI and shape of phantom is:', num_slices_phantom, num_rows_cols, num_rows_cols)

# Set display indexes
display_slice = img_slices_boundary_size + int(0.4*num_slices_phantom)
display_x = num_rows_cols // 2
display_y = num_rows_cols // 2
display_view = 0

# Generate a phantom
phantom = mbircone.phantom.gen_shepp_logan_3d(num_rows_cols, num_rows_cols, num_slices_phantom)
print('Generated phantom shape = ', np.shape(phantom))
phantom = mbircone.cone3D.pad_roi2ror(phantom, boundary_size)
print('Padded phantom shape = ', np.shape(phantom))

# display phantom
title = f'Slice {display_slice:d} of 3D Shepp Logan Phantom.'
plot_image(phantom[display_slice], title=title, filename='output/3d_shepp_logan_phantom.png', vmin=vmin, vmax=vmax)


# Generate simulated data using forward projector on the 3D shepp logan phantom.
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

# After setting the geometric parameter, the shape of the phantom is set to a fixed shape.
# Input a phantom with wrong shape will generate a bunch of issue in C.
sino = mbircone.cone3D.project(phantom, angles,
                               num_det_rows=num_det_rows, num_det_channels=num_det_channels,
                               dist_source_detector=dist_source_detector, magnification=magnification,
                               delta_pixel_detector=delta_pixel_detector, delta_pixel_image=delta_pixel_image,
                               channel_offset=channel_offset, row_offset=row_offset)

# create output folder
os.makedirs('output/3D_shepp_logan/', exist_ok=True)

print('sino shape = ', np.shape(sino), sino.dtype)
plot_image(sino[:, display_slice, :], title='sino', filename='output/3D_shepp_logan/sino-shepp-logan-3D-slice%d.png' % display_slice)
plot_gif(sino, 'output', 'sino-shepp-logan-3D')

recon = mbircone.cone3D.recon(sino, angles,
                              dist_source_detector=dist_source_detector, magnification=magnification,
                              delta_pixel_detector=delta_pixel_detector, delta_pixel_image=delta_pixel_image,
                              channel_offset=channel_offset, row_offset=row_offset,
                              sharpness=sharpness, snr_db=snr_db, max_iterations=100)

print('recon shape = ', np.shape(recon))
plot_gif(recon, 'output/3D_shepp_logan', 'recon', vmin=vmin, vmax=vmax)
plot_gif(phantom, 'output/3D_shepp_logan', 'phantom', vmin=vmin, vmax=vmax)

# Compute Normalized Root Mean Squared Error
nrmse = mbircone.phantom.nrmse(recon, phantom)

# display phantom
title = f'Slice {display_slice:d} of 3D Shepp-logan Phantom xy.'
plot_image(phantom[display_slice], title=title, filename='output/3D_shepp_logan/3d_shepp_logan_phantom_xy.png',
           vmin=vmin, vmax=vmax)
title = f'Slice {display_x:d} of 3D Shepp-logan Phantom yz.'
plot_image(phantom[:, display_x, :], title=title, filename='output/3D_shepp_logan/3d_shepp_logan_phantom_yz.png',
           vmin=vmin, vmax=vmax)
title = f'Slice {display_y:d} of 3D Shepp-logan Phantom xz.'
plot_image(phantom[:, :, display_y], title=title, filename='output/3D_shepp_logan/3d_shepp_logan_phantom_xz.png',
           vmin=vmin, vmax=vmax)


# display reconstruction
title = f'Slice {display_slice:d} of 3D Recon xy with NRMSE={nrmse:.3f}.'
plot_image(recon[display_slice], title=title, filename='output/3D_shepp_logan/3d_shepp_logan_recon_xy.png', vmin=vmin,
           vmax=vmax)
title = f'Slice {display_y:d} of 3D Recon xz with NRMSE={nrmse:.3f}.'
plot_image(recon[:, :, display_y], title=title, filename='output/3D_shepp_logan/3d_shepp_logan_recon_xz.png', vmin=vmin,
           vmax=vmax)
title = f'Slice {display_x:d} of 3D Recon yz with NRMSE={nrmse:.3f}.'
plot_image(recon[:, display_x, :], title=title, filename='output/3D_shepp_logan/3d_shepp_logan_recon_yz.png', vmin=vmin,
           vmax=vmax)

input("press Enter")
