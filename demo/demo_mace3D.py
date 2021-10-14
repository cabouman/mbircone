import os,sys
import numpy as np
import urllib.request
import tarfile
from keras.models import model_from_json
import mbircone
import display_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

"""
This file demonstrates the usage of mace3D reconstruction algorithm by downloading phantom and denoiser data from a url, generating sinogram by projecting the phantom and adding transmission noise, and finally performing a 3D MACE reconstruction.
"""
print('This file demonstrates the usage of mace3D reconstruction algorithm by downloading phantom and denoiser data from a url, generating sinogram by projecting the phantom, and finally performing a 3D MACE reconstruction.\n')

################ Define Parameters
# Geometry parameters
dist_source_detector = 839.0472
magnification = 5.572128439964856
delta_pixel_detector = 0.25
num_det_rows = 28
num_det_channels = 240
# Simulated sinogram parameters
num_views = 75
sino_noise_sigma = 0.01 # transmission noise level
# MACE recon parameters 
max_admm_itr = 10
prior_weight = 0.5


################ Download and extract data
print("Downloading and extracting data ...")
# Download phantom and params files
download_url = 'https://github.com/dyang37/mbircone_data/raw/master/demo_data.tar.gz'
urllib.request.urlretrieve(download_url, 'temp.tar.gz') 
# Extract tarball file
tar_file = tarfile.open('temp.tar.gz')
tar_file.extractall('./demo_data/')
tar_file.close()
os.remove('temp.tar.gz') 


################ Generate sinogram
print("Generating sinogram ...")
# Generate array of view angles
angles = np.linspace(0, 2*np.pi, num_views, endpoint=False)
# Generate clean sinogram by projecting phantom
phantom = np.load('./demo_data/phantom_3D.npy')
sino = mbircone.cone3D.project(phantom,angles,
            num_det_rows, num_det_channels,
            dist_source_detector, magnification,
            delta_pixel_detector=delta_pixel_detector)
# Calculate sinogram weights
weights = mbircone.cone3D.calc_weights(sino, weight_type='transmission')
# Add noise to clean sinogram
noise = sino_noise_sigma * 1./np.sqrt(weights) * np.random.normal(size=(num_views,num_det_rows,num_det_channels))
sino_noisy = sino + noise


################ Load denoiser function and model
print("Loading denoiser function and model ...")
# Load keras denoiser function
denoiser = mbircone.mace.keras_denoiser 
# Load denoiser model structure and weights
json_path = './demo_data/dncnn_params/model_dncnn/model.json'
weight_path = './demo_data/dncnn_params/model_dncnn/model.hdf5'
json_file = open(json_path, 'r')
denoiser_model_json = json_file.read() # requires keras
json_file.close()
denoiser_model = model_from_json(denoiser_model_json)
denoiser_model.load_weights(weight_path)


################ Perform MACE reconstruction
print("Performing MACE reconstruction ...")
recon_mace = mbircone.mace.mace3D(sino_noisy, angles, dist_source_detector, magnification,
        denoiser=denoiser, denoiser_args=(denoiser_model),
        max_admm_itr=max_admm_itr, prior_weight=prior_weight,
        delta_pixel_detector=delta_pixel_detector,
        weight_type='transmission')


################ Post-process Reconstruction results
print("Post processing MACE reconstruction results ...")
# Output image and results directory
output_dir = './output/mace3D/'
os.makedirs(output_dir, exist_ok=True)
# Save recon results as a numpy array
np.save(os.path.join(output_dir,"recon_mace.npy"), recon_mace)
# Plot sinogram data
display_utils.plot_image(sino_noisy[0],title='sinogram view 0, noise level=0.05',filename=os.path.join(output_dir,'sino_noisy.png'),vmin=0,vmax=4)
display_utils.plot_image(sino[0],title='clean sinogram view 0',filename=os.path.join(output_dir,'sino_clean.png'),vmin=0,vmax=4)
display_utils.plot_image(noise[0],title='sinogram additive Gaussian noise,  view 0',filename=os.path.join(output_dir,'sino_transmission_noise.png'),vmin=-0.08,vmax=0.08)
# Plot an axial slice of phantom and recon
display_utils.plot_image(phantom[1],title='phantom, axial slice 1',filename=os.path.join(output_dir,'phantom_slice.png'),vmin=0,vmax=0.5)
display_utils.plot_image(recon_mace[1],title='MACE reconstruction, axial slice 1',filename=os.path.join(output_dir,'recon_mace_slice.png'),vmin=0,vmax=0.5)
# Plot 3D phantom and recon image volumes as gif images.
display_utils.plot_gif(phantom,output_dir,'phantom',vmin=0,vmax=0.5)
display_utils.plot_gif(recon_mace,output_dir,'recon_mace',vmin=0,vmax=0.5)


input("press Enter")
