import os
import numpy as np
import mbircone
from demo_utils import plot_image, plot_gif, plt_cmp_3dobj

"""
This script is a demonstration of the 3D qGGMRF reconstruction algorithm. Demo functionality includes
 * generating a 3D Shepp Logan phantom, generating synthetic sinogram data by
 * forward projecting the phantom, performing multiple 3D qGGMRF reconstructions with variable ROR sizes and phantoms,
 * and displaying the results.
"""
print('This script is a demonstration of the 3D qGGMRF reconstruction algorithm. Demo functionality includes \
\n\t * generating a 3D Shepp Logan phantom, generating synthetic sinogram data by \
\n\t * forward projecting the phantom, performing multiple 3D qGGMRF reconstructions with variable ROR sizes and phantoms, \
\n\t * and displaying the results.')

# ###########################################################################
# Set the parameters to generate the phantom, synthetic sinogram, and do the recon
# ###########################################################################

# Change the parameters below for your own use case.

# Detector size
num_det_rows = 128
num_det_channels = 128
# Geometry parameters
magnification = 2.0                         # Ratio of (source to detector)/(source to center of rotation)
dist_source_detector = 10*num_det_channels  # distance from source to detector in ALU
# number of projection views
num_views = 64
# projection angles will be uniformly spaced within the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
# qGGMRF recon parameters
sharpness = 0.0                             # Controls regularization: larger => sharper; smaller => smoother
T = 0.0001                                     # Controls edginess of reconstruction
# display parameters
vmin = 0.10
vmax = 0.12

delta_pixel_phantom = 0.5

# Size of phantom
num_slices_phantom = 128
num_rows_phantom = 128
num_cols_phantom = 128

delta_pixel_recon = delta_pixel_phantom

# Size of recon
num_slices_recon = 128
num_rows_recon = 128
num_cols_recon = 128

# Size of recon large
num_slices_recon_large = 192
num_rows_recon_large = 192
num_cols_recon_large = 192

# Size and proportion of off-axis phantom
scale= 0.8
offset_z= 0.12

# Size of recon off-axis
num_slices_recon_off_axis = int(scale * num_slices_phantom * delta_pixel_recon / delta_pixel_phantom)
num_rows_recon_off_axis = int(scale * num_rows_phantom * delta_pixel_recon / delta_pixel_phantom)
num_cols_recon_off_axis = int(scale * num_cols_phantom * delta_pixel_recon / delta_pixel_phantom)
image_slice_offset_off_axis = offset_z * num_slices_phantom * delta_pixel_phantom

# local path to save phantom, sinogram, and reconstruction images
save_path = f'output/3D_shepp_logan_test/'
os.makedirs(save_path, exist_ok=True)

print('Genrating 3D Shepp Logan phantom ...')
######################################################################################
# Generate 3D shepp logan phantoms
######################################################################################

phantom = mbircone.phantom.gen_shepp_logan_3d(num_rows_phantom, num_cols_phantom, num_slices_phantom)
# scale the phantom by a factor of 10.0 to make the projections physical realistic -log attenuation values
phantom = phantom/10.0
print('Phantom shape = ', np.shape(phantom))

phantom_off_axis = mbircone.phantom.gen_shepp_logan_3d(num_rows_phantom, num_cols_phantom, num_slices_phantom, scale=scale, offset_z=offset_z)
phantom_off_axis = phantom_off_axis/10.0

print('Phantom off-axis shape = ', np.shape(phantom_off_axis))

######################################################################################
# Generate synthetic sinogram
######################################################################################

print('Generating synthetic sinograms ...')
sino = mbircone.cone3D.project(phantom, angles,
                               num_det_rows, num_det_channels,
                               dist_source_detector, magnification, delta_pixel_image=delta_pixel_phantom)
print('Synthetic sinogram shape: (num_views, num_det_rows, num_det_channels) = ', sino.shape)

sino_off_axis = mbircone.cone3D.project(phantom_off_axis, angles,
                               num_det_rows, num_det_channels,
                               dist_source_detector, magnification, delta_pixel_image=delta_pixel_phantom)
print('Synthetic sinogram off-axis shape: (num_views, num_det_rows, num_det_channels) = ', sino_off_axis.shape)

######################################################################################
# Perform 3D qGGMRF reconstruction
######################################################################################

print('Performing 3D qGGMRF reconstruction ...')

#recon = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification, delta_pixel_image=delta_pixel_recon, num_image_rows=num_rows_recon, num_image_cols=num_cols_recon, num_image_slices=num_slices_recon, sharpness=sharpness, T=T)
#
#print('recon shape = ', np.shape(recon))
#
#recon_large = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification, delta_pixel_image=delta_pixel_recon, num_image_rows=num_rows_recon_large, num_image_cols=num_cols_recon_large, num_image_slices=num_slices_recon_large, sharpness=sharpness, T=T)
#
#print('recon large shape = ', np.shape(recon_large))

recon_off_axis = mbircone.cone3D.recon(sino_off_axis, angles, dist_source_detector, magnification, delta_pixel_image=delta_pixel_recon, num_image_rows=num_rows_recon_off_axis, num_image_cols=num_cols_recon_off_axis, num_image_slices=num_slices_recon_off_axis, image_slice_offset=image_slice_offset_off_axis, sharpness=sharpness, T=T)

print('recon off-axis shape = ', np.shape(recon_off_axis))

#recon_pixel_pitch = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification, delta_pixel_image=delta_pixel_recon*2, num_image_rows=num_rows_recon//2, num_image_cols=num_cols_recon//2, num_image_slices=num_slices_recon//2, sharpness=sharpness, T=T)
#
#print('recon pixel-pitch shape = ', np.shape(recon_pixel_pitch))

######################################################################################
# Generate phantom, synthetic sinogram, and reconstruction images
######################################################################################
# sinogram images
for view_idx in [0, num_views//4, num_views//2]:
    view_angle = int(angles[view_idx]*180/np.pi)
    plot_image(sino[view_idx, :, :], title=f'sinogram view angle {view_angle} ',
               filename=os.path.join(save_path, f'sino-shepp-logan-3D-view_angle{view_angle}.png'))

for view_idx in [0, num_views//4, num_views//2]:
    view_angle = int(angles[view_idx]*180/np.pi)
    plot_image(sino_off_axis[view_idx, :, :], title=f'sinogram off-axis view angle {view_angle} ',
               filename=os.path.join(save_path, f'sino-off-axis-shepp-logan-3D-view_angle{view_angle}.png'))
               
# Set display indexes for phantom and recon images
display_slice_phantom = num_slices_phantom // 2
display_x_phantom = num_rows_phantom // 2
display_y_phantom = num_cols_phantom // 2

display_slice_recon = num_slices_recon // 2
display_x_recon = num_rows_recon // 2
display_y_recon = num_cols_recon // 2

display_slice_recon_large = num_slices_recon_large // 2
display_x_recon_large = num_rows_recon_large // 2
display_y_recon_large = num_cols_recon_large // 2

display_slice_phantom_off_axis = num_slices_phantom // 4
display_x_phantom_off_axis = num_rows_phantom // 2
display_y_phantom_off_axis = num_cols_phantom // 2

display_slice_recon_off_axis = num_slices_recon_off_axis // 2
display_x_recon_off_axis = num_rows_recon_off_axis // 2
display_y_recon_off_axis = num_cols_recon_off_axis // 2

display_slice_recon_pixel_pitch = num_slices_recon // 4
display_x_recon_pixel_pitch = num_rows_recon // 4
display_y_recon_pixel_pitch = num_cols_recon // 4


#plot_image(phantom[display_slice_phantom], title=f'phantom, axial slice {display_slice_phantom}',
#           filename=os.path.join(save_path, 'phantom_axial.png'), vmin=vmin, vmax=vmax)
#plot_image(phantom[:,display_x_phantom,:], title=f'phantom, coronal slice {display_x_phantom}',
#           filename=os.path.join(save_path, 'phantom_coronal.png'), vmin=vmin, vmax=vmax)
#plot_image(phantom[:,:,display_y_phantom], title=f'phantom, sagittal slice {display_y_phantom}',
#           filename=os.path.join(save_path, 'phantom_sagittal.png'), vmin=vmin, vmax=vmax)
#
#plot_image(recon[display_slice_recon], title=f'qGGMRF recon, axial slice {display_slice_recon}',
#           filename=os.path.join(save_path, 'recon_axial.png'), vmin=vmin, vmax=vmax)
#plot_image(recon[:,display_x_recon,:], title=f'qGGMRF recon, coronal slice {display_x_recon}',
#           filename=os.path.join(save_path, 'recon_coronal.png'), vmin=vmin, vmax=vmax)
#plot_image(recon[:,:,display_y_recon], title=f'qGGMRF recon, sagittal slice {display_y_recon}',
#           filename=os.path.join(save_path, 'recon_sagittal.png'), vmin=vmin, vmax=vmax)
#
#plot_image(recon_large[display_slice_recon_large], title=f'qGGMRF recon large, axial slice {display_slice_recon_large}',
#          filename=os.path.join(save_path, 'recon_large_axial.png'), vmin=vmin, vmax=vmax)
#plot_image(recon_large[:,display_x_recon_large,:], title=f'qGGMRF recon large, coronal slice {display_x_recon_large}',
#          filename=os.path.join(save_path, 'recon_large_coronal.png'), vmin=vmin, vmax=vmax)
#plot_image(recon_large[:,:,display_y_recon_large], title=f'qGGMRF recon large, sagittal slice {display_y_recon_large}',
#          filename=os.path.join(save_path, 'recon_large_sagittal.png'), vmin=vmin, vmax=vmax)

plot_image(phantom_off_axis[display_slice_phantom_off_axis], title=f'phantom off-axis, axial slice {display_slice_phantom_off_axis}',
           filename=os.path.join(save_path, 'phantom_off_axis_axial.png'), vmin=vmin, vmax=vmax)
plot_image(phantom_off_axis[:,display_x_phantom_off_axis,:], title=f'phantom off-axis, coronal slice {display_x_phantom_off_axis}',
           filename=os.path.join(save_path, 'phantom_off_axis_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(phantom_off_axis[:,:,display_y_phantom_off_axis], title=f'phantom off-axis, sagittal slice {display_y_phantom_off_axis}',
           filename=os.path.join(save_path, 'phantom_off_axis_sagittal.png'), vmin=vmin, vmax=vmax)

plot_image(recon_off_axis[display_slice_recon_off_axis], title=f'qGGMRF recon off_axis, axial slice {display_slice_recon_off_axis}',
          filename=os.path.join(save_path, 'recon_off_axis_axial.png'), vmin=vmin, vmax=vmax)
plot_image(recon_off_axis[:,display_x_recon_off_axis,:], title=f'qGGMRF recon off_axis, coronal slice {display_x_recon_off_axis}',
          filename=os.path.join(save_path, 'recon_off_axis_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(recon_off_axis[:,:,display_y_recon_off_axis], title=f'qGGMRF recon off_axis, sagittal slice {display_y_recon_off_axis}',
          filename=os.path.join(save_path, 'recon_off_axis_sagittal.png'), vmin=vmin, vmax=vmax)
          
#plot_image(recon_pixel_pitch[display_slice_recon_pixel_pitch], title=f'qGGMRF recon pixel-pitch, axial slice {display_slice_recon_pixel_pitch}',
#          filename=os.path.join(save_path, 'recon_pixel_pitch_axial.png'), vmin=vmin, vmax=vmax)
#plot_image(recon_pixel_pitch[:,display_x_recon_pixel_pitch,:], title=f'qGGMRF recon pixel-pitch, coronal slice {display_x_recon_pixel_pitch}',
#          filename=os.path.join(save_path, 'recon_pixel_pitch_coronal.png'), vmin=vmin, vmax=vmax)
#plot_image(recon_pixel_pitch[:,:,display_y_recon_pixel_pitch], title=f'qGGMRF recon pixel-pitch, sagittal slice {display_y_recon_pixel_pitch}',
#          filename=os.path.join(save_path, 'recon_pixel_pitch_sagittal.png'), vmin=vmin, vmax=vmax)
#
print(f"Images saved to {save_path}.") 
input("Press Enter")

