import os
import numpy as np
import mbircone
from demo_utils import plot_image, plot_gif

"""
The purpose of this script is to test the \'project\' method in cone3D.py. This script generates a small-scale
 * 9x9 point-spread phantom, and generates a 9x9x64 sinogram. The parameter detector_row_offset
 * is the stepped from 0.0 to 1.0 by increments of 0.1, and the experiment is repeated for each
 * increment. Three view angles are printed for each sinogram.
"""

print('The purpose of this script is to test the \'project\' method in cone3D.py. This script generates a small-scale \
\n\t * 9x9 point-spread phantom, and generates a 9x9x64 sinogram. The parameter detector_row_offset \
\n\t * is the stepped from 0.0 to 1.0 by increments of 0.1, and the experiment is repeated for each \
\n\t * increment. Three view angles are printed for each sinogram.')

# ##################################################################
# Set the parameters to generate the phantom and synthetic sinogram
# ##################################################################

# Change the parameters below for your own use case.

# Detector size
num_det_rows = 9
num_det_channels = 9
# Geometry parameters
magnification = 1.0  # Ratio of (source to detector)/(source to center of rotation)
dist_source_detector = 90  # distance from source to detector in ALU
# number of projection views
num_views = 64
# projection angles will be uniformly spaced within the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

# display parameters
vmin = 0.10
vmax = 0.12

# Size of phantom
num_slices_phantom = 9
num_rows_phantom = 9
num_cols_phantom = 9
delta_pixel_phantom = 1

# local path to save phantom, sinogram, and reconstruction images
save_path = f'output/test_forward_proj/'
os.makedirs(save_path, exist_ok=True)

######################################################################################
# Generate 9x9 point-spread phantom
######################################################################################

print('Genrating 9x9 point-spread phantom ...')

phantom = np.zeros((num_slices_phantom, num_rows_phantom, num_cols_phantom))
phantom[4, 4, 4] = 1

######################################################################################
# Generate synthetic sinogram
######################################################################################

print('Generating synthetic sinograms ...')

# Generate detector offset values
start = 0.0
stop = 1.0
step = 0.1
for offset in np.arange(start, stop + step, step):
    sino = mbircone.cone3D.project(phantom, angles,
                                   num_det_rows, num_det_channels,
                                   dist_source_detector, magnification, det_row_offset=offset,
                                   delta_pixel_image=delta_pixel_phantom)
    print('Synthetic sinogram shape: (num_views, num_det_rows, num_det_channels) = ', sino.shape)

    # Generate sinogram images
    for view_idx in [0, num_views // 4, num_views // 2]:
        view_angle = int(angles[view_idx] * 180 / np.pi)
        plot_image(sino[view_idx, :, :], title=f'sinogram view angle {view_angle}, det_row_offset {offset}',
                   filename=os.path.join(save_path, f'sino-shepp-logan-3D-offset-{offset}-view-angle-{view_angle}.png'))

# Set display indexes for phantom
display_slice_phantom = num_slices_phantom // 2
display_x_phantom = num_rows_phantom // 2
display_y_phantom = num_cols_phantom // 2

# Display phantom images
plot_image(phantom[display_slice_phantom], title=f'phantom, axial slice {display_slice_phantom}',
           filename=os.path.join(save_path, 'phantom_axial.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:, display_x_phantom, :], title=f'phantom, coronal slice {display_x_phantom}',
           filename=os.path.join(save_path, 'phantom_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:, :, display_y_phantom], title=f'phantom, sagittal slice {display_y_phantom}',
           filename=os.path.join(save_path, 'phantom_sagittal.png'), vmin=vmin, vmax=vmax)

print(f"Images saved to {save_path}.")
input("Press Enter")

