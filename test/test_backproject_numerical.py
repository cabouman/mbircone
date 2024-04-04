import os
import numpy as np
import mbircone
from test_utils import plot_image, plot_gif, nrmse
from scipy.io import savemat

print('This script tests the back projector by numerically showing that <y, Ax> = <x, A^t y>. Demo includes:\
\n\t * Generating a random image x;\
\n\t * Generating a random sinogram y;\
\n\t * Computing <y, Ax> with the forward projector;\
\n\t * Computing <x, A^t y> with the back projector;\
\n\t * Calculating the NRMSE between <y, Ax> and <x, A^t y>.\n')

# ###########################################################################
# Set the parameters for the scanner geometry
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

######################################################################################
# Generate a random phantom with uniform distribution over [0,1)
######################################################################################
print('Generating x: a random phantom with uniform distribution over [0,1) ...\n')
x = np.random.rand(num_phantom_slices, num_phantom_rows, num_phantom_cols)
print("Shape of x = ", x.shape)

######################################################################################
# Generate a random sinogram with uniform distribution over [0,1)
######################################################################################
print('Generating y: a random sinogram with uniform distribution over [0,1) ...\n')
y = np.random.rand(num_views, num_det_rows, num_det_channels)
print("Shape of y = ", y.shape)

######################################################################################
# Compute <y, Ax>
######################################################################################
# Computing Ax with forward projector ...
Ax = mbircone.cone3D.project(x, angles,
                             num_det_rows, num_det_channels,
                             dist_source_detector, magnification)
print('Computing <y, Ax> ...\n')
y_Ax_inner_product = np.sum(np.multiply(y, Ax))
print("<y, Ax> = ", y_Ax_inner_product)


######################################################################################
# Compute <x, A^t y>
######################################################################################
# Computing A^t y with back projector ...
Aty = mbircone.cone3D.backproject(y, angles, dist_source_detector, magnification)
print('Computing <x, A^t y> ...\n')
x_Aty_inner_product = np.sum(np.multiply(x, Aty))
print("<x, Aty> = ", x_Aty_inner_product)

print("nrmse(<y, Ax>, <x, A^t y>) = ", nrmse(y_Ax_inner_product, x_Aty_inner_product))
