import os,sys
import numpy as np
import urllib.request
import tarfile
from keras.models import model_from_json
import mbircone
import display_utils, denoiser_utils
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
# path to downloaded files. Please change them accordingly if you replace any of them with your own files.
phantom_path = './demo_data/phantom_3D.npy' # 3D image volume phantom file
json_path = './demo_data/dncnn_params/model_dncnn/model.json' # model architecture file
weight_path = './demo_data/dncnn_params/model_dncnn/model.hdf5' # model weight file

################ Download and extract data
# check if demo_data folder is already there.
is_download = True
if os.path.exists('./demo_data/'):
    is_download = display_utils.query_yes_no("demo_data folder already exists. Do you still want to download and overwrite the files?")
if is_download:
    # download the data from url.
    print("Downloading and extracting data ...")
    # Download phantom and cnn denoiser params files.
    # A tarball file will be downloaded from the given url and extracted to demo_data/ folder.
    # the tarball file contains the following files:
    # an image volume phantom file phantom_3D.npy. You can replace this file with your own phantom data.
    # dncnn parameter files stored in demo_data/dncnn_params/ directory
    download_url = 'https://github.com/dyang37/mbircone_data/raw/master/demo_data.tar.gz'
    try:
        urllib.request.urlretrieve(download_url, 'temp.tar.gz') 
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise RuntimeError(f'HTTP status code {e.code}: URL authentication failed! Currently we do not support downloading data from a url that requires authentication.')
        elif e.code == 403:
            raise RuntimeError(f'HTTP status code {e.code}: URL forbidden! Please make sure the provided URL is public.')
        elif e.code == 404:
            raise RuntimeError(f'HTTP status code {e.code}: URL not Found! Please check and make sure the download URL provided is correct.')
        else:
            raise RuntimeError(f'HTTP status code {e.code}: {e.reason}. For more details please refer to https://en.wikipedia.org/wiki/List_of_HTTP_status_codes')
    except urllib.error.URLError as e:
        raise RuntimeError('URLError raised! Please check your internet connection.')
    
    # Extract tarball file
    tar_file = tarfile.open('temp.tar.gz')
    tar_file.extractall('./demo_data/')
    tar_file.close()
    os.remove('temp.tar.gz') 
    input("Data download and extraction finished. Press Enter to continue.")
else:
    print("Skipped data download and extraction step.")

################ Generate sinogram
print("Generating sinogram ...")
# Generate array of view angles
angles = np.linspace(0, 2*np.pi, num_views, endpoint=False)
# Generate clean sinogram by projecting phantom
phantom = np.load(phantom_path)
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
# Load cnn denoiser function
denoiser = denoiser_utils.cnn_denoiser 
# Load denoiser model structure and weights
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
