import os
import numpy as np
import mbircone
from demo_utils import plot_image, plot_gif, plt_cmp_3dobj

"""
This script is a demonstration of the 3D qGGMRF reconstruction algorithm. Demo functionality includes
 * generating a 3D Shepp Logan phantom, and zero-padding it to the size of ROR
 * generating synthetic sinogram data by forward projecting the phantom
 * performing a 3D qGGMRF reconstructions and displaying the results.
"""
print('This script is a demonstration of the 3D qGGMRF reconstruction algorithm. Demo functionality includes \
\n\t * generating a 3D Shepp Logan phantom, and zero-padding it to the size of ROR \
\n\t * generating synthetic sinogram data by forward projecting the phantom \
\n\t * performing a 3D qGGMRF reconstructions and displaying the results.')

# ###########################################################################
# Set the parameters to generate the phantom, synthetic sinogram, and do the recon
# ###########################################################################

# Change the parameters below for your own use case.

# detector size
num_det_rows = 256
num_det_channels = 256
# Geometry parameters
magnification = 2.0                         # Ratio of (source to detector)/(source to center of rotation)
dist_source_detector = 10*num_det_channels  # distance from source to detector in ALU
# number of projection views
num_views = 64
# projection angles will be uniformly spaced within the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
# qGGMRF recon parameters
sharpness = 0.0                             # Controls regularization: larger => sharper; smaller => smoother
T = 0.1                                     # Controls edginess of reconstruction
# display parameters
vmin = 0.10
vmax = 0.12

# local path to save phantom, sinogram, and reconstruction images
save_path = f'output/3D_shepp_logan_num_rows_cols_140__slice_offset/'
os.makedirs(save_path, exist_ok=True)


print('Genrating 3D Shepp Logan phantom ...')
######################################################################################
# This section determines the phantom size corresponding to the geometry parameters
######################################################################################
# Compute ROR (Region of Reconstruction) and boundary size
(num_slices_ROR, num_rows_ROR, num_cols_ROR), boundary_size = mbircone.cone3D.compute_img_size(num_views, num_det_rows, num_det_channels, dist_source_detector, magnification)
print(f'ROR size = ({num_slices_ROR}, {num_rows_ROR}, {num_cols_ROR})')

######################################################################################
# Generate a 3D shepp logan phantom with size calculated in previous section
######################################################################################
num_slices_ROI, num_rows_ROI, num_cols_ROI = num_slices_ROR//2, num_rows_ROR//2, num_cols_ROR//2
boundary_size = ((num_slices_ROR-num_slices_ROI)//2, (num_rows_ROR-num_rows_ROI)//2, (num_cols_ROR-num_cols_ROI)//2)
# generate phantom within ROI
phantom = mbircone.phantom.gen_shepp_logan_3d(num_rows_ROI, num_cols_ROI, num_slices_ROI)       
print(f'phantom ROI size = ', phantom.shape)
# zero-pad phantom to ROR
phantom = mbircone.cone3D.pad_roi2ror(phantom, boundary_size)                                   
phantom = np.pad(phantom, ((0,num_slices_ROR-phantom.shape[0]),(0,num_rows_ROR-phantom.shape[1]),(0,num_cols_ROR-phantom.shape[2]))) 
# scale the phantom by a factor of 10.0 to make the projections physical realistic -log attenuation values
phantom = phantom/10.0
print('Padded ROR phantom shape = ', np.shape(phantom))


######################################################################################
# Generate synthetic sinogram
######################################################################################
print('Generating synthetic sinogram ...')
sino = mbircone.cone3D.project(phantom, angles, 
                               num_det_rows, num_det_channels, 
                               dist_source_detector, magnification)
print('Synthetic sinogram shape: (num_views, num_det_rows, num_det_channels) = ', sino.shape)


######################################################################################
# Perform 3D qGGMRF reconstruction
######################################################################################
print('Performing 3D qGGMRF reconstruction ...')
delta_pixel_image = 1./magnification
#num_slices = 140
#num_rows=140
#num_cols=130
delta_pixel_image = 1.0/magnification
#ror_radius = 70*delta_pixel_image
num_rows = 140
num_cols = 140
num_slices = 190
slice_offset = 40*delta_pixel_image
recon = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification, sharpness=sharpness, T=T, 
                              #ror_radius=ror_radius, 
                              num_rows=num_rows, num_cols=num_cols, num_slices=num_slices, slice_offset=slice_offset)
num_slices_recon, num_rows_recon, num_cols_recon = recon.shape
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
display_slice = num_slices_ROR // 2
display_x = num_rows_ROR // 2
display_y = num_cols_ROR // 2
# phantom images
plot_image(phantom[display_slice], title=f'phantom, axial slice {display_slice}',
           filename=os.path.join(save_path, 'phantom_axial.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,display_x,:], title=f'phantom, coronal slice {display_x}',
           filename=os.path.join(save_path, 'phantom_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,:,display_y], title=f'phantom, sagittal slice {display_y}', 
           filename=os.path.join(save_path, 'phantom_sagittal.png'), vmin=vmin, vmax=vmax)
# Set display indexes for recon images
display_slice = num_slices_recon // 2
display_x = num_rows_recon // 2
display_y = num_cols_recon // 2
# recon images
plot_image(recon[display_slice], title=f'qGGMRF recon, axial slice {display_slice}',
           filename=os.path.join(save_path, 'recon_axial.png'), vmin=vmin, vmax=vmax)
plot_image(recon[:,display_x,:], title=f'qGGMRF recon, coronal slice {display_x}', 
           filename=os.path.join(save_path, 'recon_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(recon[:,:,display_y], title=f'qGGMRF recon, sagittal slice {display_y}', 
           filename=os.path.join(save_path, 'recon_sagittal.png'), vmin=vmin, vmax=vmax)
print(f"Images saved to {save_path}.") 
input("Press Enter")
