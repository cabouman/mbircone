import os
import numpy as np
import mbircone
from demo_utils import plot_image, plot_gif, plt_cmp_3dobj

# Set sinogram shape
num_det_rows = 200
num_det_channels = 128
num_views = 144

# Reconstruction parameters
sharpness = 0.2
snr_db = 31.0

# magnification is unitless.
magnification = 2

# All distances are in unit of 1 ALU = 1 mm.
dist_source_detector = 600
delta_pixel_detector = 0.9
delta_pixel_image = 1
channel_offset = 0
row_offset = 0
max_iterations = 100

# Display parameters
vmin = 1.0
vmax = 1.1
filename = 'output/3D_shepp_logan/results.png'

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

# Set phantom parameters to generate a phantom inside ROI according to ROR and boundary_size.
# All valid pixels should be inside ROI.
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

# Generate simulated data using forward projector on the 3D shepp logan phantom.
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

# After setting the geometric parameter, the shape of the input phantom should be equal to the calculated geometric parameter.
# Input a phantom with wrong shape will generate a bunch of issue in C.
sino = mbircone.cone3D.project(phantom, angles,
                               num_det_rows=num_det_rows, num_det_channels=num_det_channels,
                               dist_source_detector=dist_source_detector, magnification=magnification,
                               delta_pixel_detector=delta_pixel_detector, delta_pixel_image=delta_pixel_image,
                               channel_offset=channel_offset, row_offset=row_offset)

# create output folder
os.makedirs('output/3D_shepp_logan/', exist_ok=True)

print('sino shape = ', np.shape(sino), sino.dtype)
plot_image(sino[display_view, :, :], title='sino', filename='output/3D_shepp_logan/sino-shepp-logan-3D-view(%.2f).png' % angles[0])
# plot_gif(sino, 'output', 'sino-shepp-logan-3D')

#weights = mbircone.cone3D.calc_weights(sino, weight_type='transmission')
weights = np.zeros(np.shape(sino))

recon = mbircone.cone3D.recon(sino, angles,
                              dist_source_detector=dist_source_detector, magnification=magnification,
                              delta_pixel_detector=delta_pixel_detector, delta_pixel_image=delta_pixel_image,
                              channel_offset=channel_offset, row_offset=row_offset,
                              sharpness=sharpness, snr_db=snr_db, max_iterations=max_iterations,
                              weights=weights)

print('recon shape = ', np.shape(recon))
np.save('output/3D_shepp_logan/recon.npy', recon)

#Display and compare reconstruction
plt_cmp_3dobj(phantom, recon, display_slice, display_x, display_y, vmin, vmax, filename)

input("press Enter")
