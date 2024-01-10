import os
import numpy as np
import h5py
import mbircone
from demo_utils import plot_image, plot_gif

"""
This script demonstrates the utility function of writing a image to an HDF5 file. 
Demo functionality includes:
 * Generating a 3D Shepp Logan phantom;
 * Writing the image data as a HDF5 file using function `mbircone.utils.hdf5_write`;
 * Printing out the image metadata and displaying the image slices.
"""
print('This script demonstrates how to save image data in HDF5 format.\
Demo functionality includes:\
\n\t * Generating a 3D Shepp Logan phantom; \
\n\t * Writing the image data as a HDF5 file using function `mbircone.utils.hdf5_write`; \
\n\t * Printing out the image metadata and displaying the image slices.\n')

######### local path to save the HDF5 file and image slices
save_path = f'output/HDF5_IO/'
os.makedirs(save_path, exist_ok=True)
# path to the HDF5 file
hdf5_filename = os.path.join(save_path, "phantom.h5") 

######### dimension of the phantom
num_phantom_slices = 128
num_phantom_rows = 128
num_phantom_cols = 128

######### Synthetic image metadata
recon_description = "3D shepp logan phantom" # description of the image data
alu_description = "1 ALU = 5 mm" # Description of arbitrary length unit (ALU)
delta_pixel_image = 1 # image pixel spacing = 1 ALU

# Set display parameters for Shepp Logan phantom
vmin = 1.0
vmax = 1.2

print('Genrating 3D Shepp Logan phantom ...\n')
######################################################################################
# Generate a 3D shepp logan phantom
######################################################################################
phantom = mbircone.phantom.gen_shepp_logan_3d(num_phantom_rows, num_phantom_cols, num_phantom_slices)
print('Phantom shape = ', np.shape(phantom))

######################################################################################
# Save the phantom data as an HDF5 file
######################################################################################
print('Saving phantom data as an HDF5 file ...\n')
mbircone.utils.hdf5_write(phantom, filename=hdf5_filename, 
                          recon_description=recon_description, alu_description=alu_description, delta_pixel_image=delta_pixel_image)

######################################################################################
# Display phantom slices
######################################################################################
# Set display indexes for phantom and recon images
display_slice_phantom = num_phantom_slices // 2
display_x_phantom = num_phantom_rows // 2
display_y_phantom = num_phantom_cols // 2

plot_image(phantom[display_slice_phantom], title=f'phantom, axial slice {display_slice_phantom}',
           filename=os.path.join(save_path, 'phantom_axial.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,display_x_phantom,:], title=f'phantom, coronal slice {display_x_phantom}',
           filename=os.path.join(save_path, 'phantom_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,:,display_y_phantom], title=f'phantom, sagittal slice {display_y_phantom}',
           filename=os.path.join(save_path, 'phantom_sagittal.png'), vmin=vmin, vmax=vmax)
           
print(f"Images saved to {save_path}.") 
input("Press Enter")

