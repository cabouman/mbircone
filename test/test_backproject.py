import os
import numpy as np
import mbircone
from test_utils import plot_image, plot_gif
from scipy.io import savemat

"""
This script tests the functionality of the back projector. Demo includes:
 * Generating a cubic phantom;
 * Forward projecting the phantoma to form synthetic sinogram data;
 * Back projecting the sinogram;
 * Displaying the phantom and back projection results.
"""

print('This script tests the functionality of the back projector. Demo includes:\
\n\t * Generating a cubic phantom;\
\n\t * Forward projecting the phantom to form synthetic sinogram data;\
\n\t * Back projecting the sinograms;\
\n\t * Displaying the phantom and back projection results.\n')

# ###########################################################################
# Set the parameters to generate the phantom, synthetic sinogram, and do the recon
# ###########################################################################

# Change the parameters below for your own use case.

# Detector and geometry parameters
num_det_rows = 128                           # Number of detector rows
num_det_channels = 128                       # Number of detector channels
magnification = 2.0                          # Ratio of (source to detector)/(source to center of rotation)
dist_source_detector = 3.0*num_det_channels  # Distance from source to detector in ALU
num_views = 128                               # Number of projection views

# Generate uniformly spaced view angles in the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

# Set phantom generation parameters
num_phantom_slices = num_det_rows           # Set number of phantom slices = to the number of detector rows
num_phantom_rows = num_det_channels         # Make number of phantom rows and columns = to number of detector columns
num_phantom_cols = num_det_channels

# local path to save phantom, sinogram, and reconstruction images
save_path = f'output/test_backprojector_cubic_phantom/'
os.makedirs(save_path, exist_ok=True)

######################################################################################
# Generate a cubic phantom at the the top left of field of view
######################################################################################
print('Genrating a cubic phantom at the top left of field of view ...\n')
phantom = np.zeros((num_phantom_slices, num_phantom_rows, num_phantom_cols))
phantom[num_phantom_slices*2//8:num_phantom_slices*4//8, num_phantom_rows*2//8:num_phantom_rows*4//8, num_phantom_cols*2//8:num_phantom_cols*4//8] = 1
print('Phantom shape = ', np.shape(phantom))

######################################################################################
# Generate synthetic sinogram
######################################################################################
print('Generating synthetic sinogram ...\n')
sino = mbircone.cone3D.project(phantom, angles,
                               num_det_rows, num_det_channels,
                               dist_source_detector, magnification)
print('Synthetic sinogram shape: (num_views, num_det_rows, num_det_channels) = ', sino.shape)


######################################################################################
# Perform back projection
######################################################################################
print('Back projecting the sinogram ...\n')
backprojection = mbircone.cone3D.backproject(sino, angles, dist_source_detector, magnification)
print('backprojection shape = ', np.shape(backprojection))

######################################################################################
# Display phantom, synthetic sinogram, and reconstruction images
######################################################################################
# Set display indexes for phantom and recon images
display_slice_phantom = num_phantom_slices *3//8
display_x_phantom = num_phantom_rows *3//8
display_y_phantom = num_phantom_cols *3//8

# sinogram images
for view_idx in [0, num_views//4, num_views//2]:
    view_angle = int(angles[view_idx]*180/np.pi)
    plot_image(sino[view_idx, :, :], title=f'sinogram view angle {view_angle} ',
               filename=os.path.join(save_path, f'sino-shepp-logan-3D-view_angle{view_angle}.png'))

# display phantom images
plot_image(phantom[display_slice_phantom], title=f'phantom, axial slice {display_slice_phantom}',
           filename=os.path.join(save_path, 'phantom_axial.png'))
plot_image(phantom[:,display_x_phantom,:], title=f'phantom, coronal slice {display_x_phantom}',
           filename=os.path.join(save_path, 'phantom_coronal.png'))
plot_image(phantom[:,:,display_y_phantom], title=f'phantom, sagittal slice {display_y_phantom}',
           filename=os.path.join(save_path, 'phantom_sagittal.png'))

# display backprojection images
plot_image(backprojection[display_slice_phantom], title=f'backprojection, axial slice {display_slice_phantom}',
           filename=os.path.join(save_path, 'backprojection_axial.png'))
plot_image(backprojection[:,display_x_phantom,:], title=f'backprojection, coronal slice {display_x_phantom}',
           filename=os.path.join(save_path, 'backprojection_coronal.png'))
plot_image(backprojection[:,:,display_y_phantom], title=f'backprojection, sagittal slice {display_y_phantom}',
           filename=os.path.join(save_path, 'backprojection_sagittal.png'))
          
print(f"Images saved to {save_path}.") 
input("Press Enter")
