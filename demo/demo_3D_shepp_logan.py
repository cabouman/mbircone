import os
import numpy as np
import mbircone
from demo_utils import plot_image

# Set phantom Shape
num_rows_cols = 65  # Assumes a square image
num_slices_phantom = 80

# Set sinogram shape
num_slices_sino = 64
num_channels_sino = 64
num_views = 120

# Set display indexes
display_slice = 32
display_x = num_rows_cols // 2
display_y = num_rows_cols // 2

# Reconstruction parameters
sharpness = 2
snr_db = 30.0

# magnification is unitless.
magnification = 8.651889263

# All distances are in unit of 1 ALU = 1 mm.
dist_source_detector = 619.7902
delta_pixel_detector = 3.2
channel_offset = -0.9384
row_offset = 0.5654

# Display parameters
vmin = 1.0
vmax = 1.1

# Generate a phantom
# phantom = mbircone.phantom.gen_shepp_logan_3d(num_rows_cols, num_rows_cols, num_slices_phantom)

phantom = mbircone.phantom.gen_shepp_logan_3d(num_rows_cols, num_rows_cols, num_slices_sino)
total_pad = num_slices_phantom - num_slices_sino
left_pad = total_pad // 2
phantom = np.pad(phantom, ((left_pad, total_pad - left_pad), (0, 0), (0, 0)), 'constant', constant_values=0)

print('phantom shape = ', np.shape(phantom))
# display phantom
title = f'Slice {display_slice:d} of 3D Shepp Logan Phantom.'
plot_image(phantom[display_slice], title=title, filename='output/3d_shepp_logan_phantom.png', vmin=vmin, vmax=vmax)

print("phantom min max = ", np.min(phantom), np.max(phantom))

angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

# After setting the geometric parameter, the shape of the phantom is set to a fixed shape.
# Input a phantom with wrong shape will generate a bunch of issue in C.
sino = mbircone.cone3D.project(angles, phantom,
                               num_slices=num_slices_sino, num_channels=num_channels_sino,
                               dist_source_detector=dist_source_detector, magnification=magnification,
                               delta_pixel_detector=delta_pixel_detector,
                               channel_offset=channel_offset, row_offset=row_offset)

print('sino shape = ', np.shape(sino), sino.dtype)
plot_image(sino[:, display_slice, :], title='sino', filename='output/sino-shepp-logan-3D.png')

recon = mbircone.cone3D.recon(sino, angles,
                              dist_source_detector=dist_source_detector, magnification=magnification,
                              delta_pixel_detector=delta_pixel_detector,
                              channel_offset=channel_offset, row_offset=row_offset,
                              sharpness=sharpness, snr_db=snr_db)

print('recon shape = ', np.shape(recon))
# Compute Normalized Root Mean Squared Error
# nrmse = mbircone.phantom.nrmse(recon, phantom)

# create output folder
os.makedirs('output/3D_shepp_logan/', exist_ok=True)

# display fix resolution reconstruction
title = f'Slice {display_slice:d} of 3D Recon xy.'
plot_image(recon[display_slice], title=title, filename='output/3D_shepp_logan/3d_shepp_logan_recon_xy.png', vmin=vmin,
           vmax=vmax)

# display fix resolution reconstruction
title = f'Slice {display_y:d} of 3D Recon xz.'
plot_image(recon[:, :, display_y], title=title, filename='output/3D_shepp_logan/3d_shepp_logan_recon_xz.png', vmin=vmin,
           vmax=vmax)

# display fix resolution reconstruction
title = f'Slice {display_x:d} of 3D Recon yz.'
plot_image(recon[:, display_x, :], title=title, filename='output/3D_shepp_logan/3d_shepp_logan_recon_yz.png', vmin=vmin,
           vmax=vmax)

# display fix resolution reconstruction
title = f'Slice {display_slice:d} of 3D Recon xy.'
plot_image(phantom[display_slice], title=title, filename='output/3D_shepp_logan/3d_shepp_logan_phantom_xy.png',
           vmin=vmin, vmax=vmax)

# display fix resolution reconstruction
title = f'Slice {display_y:d} of 3D Recon xz.'
plot_image(phantom[:, :, display_y], title=title, filename='output/3D_shepp_logan/3d_shepp_logan_phantom_xz.png',
           vmin=vmin, vmax=vmax)

# display fix resolution reconstruction
title = f'Slice {display_x:d} of 3D Recon yz.'
plot_image(phantom[:, display_x, :], title=title, filename='output/3D_shepp_logan/3d_shepp_logan_phantom_yz.png',
           vmin=vmin, vmax=vmax)

input("press Enter")
