import os, sys
import numpy as np
import math
import urllib.request
import tarfile
from keras.models import model_from_json
import argparse
import getpass
from psutil import cpu_count
import mbircone
import demo_utils, denoiser_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
This script is a quick demonstration of the mace4D reconstruction algorithm.  Demo functionality includes
 * downloading phantom and denoiser data from specified urls
 * downsampling the phantom along all three dimensions
 * generating sinogram by projecting the phantom and then adding transmission noise
 * performing a 3D MACE reconstruction.
"""
print('This script is a quick demonstration of the mace4D reconstruction algorithm.  Demo functionality includes \
\n\t * downloading phantom and denoiser data from specified urls \
\n\t * downsampling the phantom along all three dimensions \
\n\t * generating sinogram by projecting the phantom and then adding transmission noise\
\n\t * performing a 4D MACE reconstruction.')

# ###########################################################################
# Set the parameters to get the data and do the recon 
# ###########################################################################

# Change the parameters below for your own use case.

# Denoiser function to be used in MACE. For the built-in demo, this should be one of dncnn_keras or dncnn_ct
# Other denoisers built in keras can be used with minimal modification by putting the architecture and weights
# in model.json and model.hdf5 in the denoiser_path set below
denoiser_type = 'dncnn_ct'

# The url to the data repo.
data_repo_url = 'https://github.com/cabouman/mbir_data/raw/master/'

# Download url to the index file.
# This file will be used to retrieve urls to files that we are going to download
yaml_url = os.path.join(data_repo_url, 'index.yaml')

# Choice of phantom and denoiser files. 
# These should be valid choices specified in the index file. 
# The urls to phantom data and NN weights will be parsed from data_repo_url and the choices of phantom and denoiser specified below.
phantom_name = 'bottle_cap_4D'
denoiser_name = denoiser_type

# destination path to download and extract the phantom and NN weight files.
target_dir = './demo_data/'   
# path to store output recon images
output_dir = './output/mace4D_fast/'  

# Geometry parameters
dist_source_detector = 839.0472     # Distance between the X-ray source and the detector in units of ALU
magnification = 5.572128439964856   # magnification = (source to detector distance)/(source to center-of-rotation distance)
delta_pixel_detector = 0.25         # Scalar value of detector pixel spacing in units of ALU
num_det_rows = 14                   # number of detector rows
num_det_channels = 120              # number of detector channels

# Simulated sinogram parameters
num_views = 75               # number of projection views
sino_noise_sigma = 0.01      # transmission noise level

# MACE recon parameters
max_admm_itr = 10            # max ADMM iterations for MACE reconstruction
prior_weight = 0.5           # cumulative weights for three prior agents.
# ######### End of parameters #########


# ###########################################################################
# Download and extract data 
# ###########################################################################

# Download the url index file and return path to local file. 
index_path = demo_utils.download_and_extract(yaml_url, target_dir) 
# Load the url index file as a directionary
url_index = demo_utils.load_yaml(index_path)
# get urls to phantom and denoiser parameter file
phantom_url = os.path.join(data_repo_url, url_index['phantom'][phantom_name])  # url to download the 3D image volume phantom file
denoiser_url = os.path.join(data_repo_url, url_index['denoiser'][denoiser_name])  # url to download the denoiser parameter file 

# download phantom file
phantom_path = demo_utils.download_and_extract(phantom_url, target_dir)
# download and extract NN weights and structure files
denoiser_path = demo_utils.download_and_extract(denoiser_url, target_dir)


# ###########################################################################
# Get cluster object for multi-node computation
# ###########################################################################
parser = argparse.ArgumentParser(description='Get configs path')
parser.add_argument('--configs_path', type=str, default=None, help="Configs path")
args = parser.parse_args()
if args.configs_path is None:
    num_cpus = cpu_count(logical=False)
    if num_cpus >= 4:
        num_worker_per_node = int(np.sqrt(num_cpus))
    else:
        num_worker_per_node = num_cpus
    cluster, maximum_possible_nb_worker = mbircone.parallel_utils.get_cluster_ticket(
        'LocalHost',
        num_worker_per_node=num_worker_per_node)
    num_threads = num_cpus // num_worker_per_node

else:
    # Load cluster setup parameter.
    configs = demo_utils.load_yaml(args.configs_path)
    # Set openmp number of threads
    num_threads = configs['cluster_params']['num_threads_per_worker']

    if configs['job_queue_sys'] == 'LocalHost':
        cluster, maximum_possible_nb_worker = mbircone.parallel_utils.get_cluster_ticket(
            configs['job_queue_sys'],
            num_worker_per_node=configs['cluster_params']['num_worker_per_node'])
    else:
        cluster, maximum_possible_nb_worker = mbircone.parallel_utils.get_cluster_ticket(
            job_queue_sys=configs['job_queue_sys'],
            num_worker_per_node=configs['cluster_params']['num_worker_per_node'],
            num_nodes=configs['cluster_params']['num_nodes'],
            num_threads_per_worker=configs['cluster_params']['num_threads_per_worker'],
            maximum_allowable_walltime=configs['cluster_params']['maximum_allowable_walltime'],
            maximum_memory_per_node=configs['cluster_params']['maximum_memory_per_node'],
            death_timeout=configs['cluster_params']['death_timeout'],
            infiniband_flag=configs['cluster_params']['infiniband_flag'],
            par_env=configs['cluster_params']['par_env'],
            queue_sys_opt=configs['cluster_params']['queue_sys_opt'],
            local_directory=configs['cluster_params']['local_directory'].replace('$USER', getpass.getuser()),
            log_directory=configs['cluster_params']['log_directory'].replace('$USER', getpass.getuser()))
print(cluster)


# ###########################################################################
# Generate downsampled phantom 
# ###########################################################################
print("Generating downsampled 3D phantom volume ...")

# load original phantom
phantom_orig = np.load(phantom_path)
print("shape of original phantom = ", phantom_orig.shape)

# downsample the original phantom along slice axis
(Nt, Nz, Nx, Ny) = phantom_orig.shape
Nx_ds = Nx // 2 + 1
Ny_ds = Ny // 2 + 1
Nz_ds = Nz // 2
phantom = np.array([demo_utils.image_resize(phantom_orig[t], (Nx_ds, Ny_ds)) for t in range(Nt)])

# Take first half of the slices to form the downsampled phantom.
phantom = phantom[:,:Nz_ds,:,:]
print("shape of downsampled phantom = ", phantom.shape)


# ###########################################################################
# Generate sinogram
# ###########################################################################
print("Generating sinogram ...")

# Generate view angles and sinogram with weights
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
sino = np.array([mbircone.cone3D.project(phantom[t], angles,
                               num_det_rows, num_det_channels,
                               dist_source_detector, magnification,
                               delta_pixel_detector=delta_pixel_detector) for t in range(Nt)])
weights = mbircone.cone3D.calc_weights(sino, weight_type='transmission')

# Add transmission noise
noise = sino_noise_sigma * 1. / np.sqrt(weights) * np.random.normal(size=(Nt, num_views, num_det_rows, num_det_channels))
sino_noisy = sino + noise


# ###########################################################################
# Set up the denoiser
# ###########################################################################
# This demo includes a custom CNN trained on CT images and a generic DnCNN in keras 
# The choice is set in the parameter section above.
print("Loading denoiser function and model ...")

# DnCNN denoiser trained on CT images. This is the denoiser that we recommend using.
if denoiser_type == 'dncnn_ct':
    print("Denoiser function: custom DnCNN trained on CT images.")

    # Load denoiser model structure and weights
    denoiser_model = denoiser_utils.DenoiserCT(checkpoint_dir=os.path.join(denoiser_path, 'model_dncnn_ct'))

    # Define the denoiser using this model.  This version requires some interface code to match with MACE.
    def denoiser(img_noisy):
        testData_obj = denoiser_utils.DataLoader(img_noisy)
        denoiser_model.denoise(testData_obj)
        img_denoised = np.stack(testData_obj.outData, axis=0)
        return np.squeeze(img_denoised)

# DnCNN denoiser in Keras. This denoiser model is trained on natural images. 
elif denoiser_type == 'dncnn_keras':
    print("Denoiser function: DnCNN trained on natural images.")

    # Load denoiser model structure and weights
    json_path = os.path.join(denoiser_path, 'model_dncnn_keras/model.json')  # model architecture file
    weight_path = os.path.join(denoiser_path, 'model_dncnn_keras/model.hdf5')  # model weight file
    with open(json_path, 'r') as json_file:
        denoiser_model = model_from_json(json_file.read())  # load model architecture

    denoiser_model.load_weights(weight_path)  # load model weights

    # Define the denoiser using this model.
    def denoiser(img_noisy):
        img_denoised = denoiser_model.predict(img_noisy)  # inference
        return np.squeeze(img_denoised)
else:
    raise RuntimeError('Unkown denoiser_type. Should be either dncnn_ct or dncnn_keras.')


# ###########################################################################
# Perform MACE reconstruction
# ###########################################################################
print("Performing MACE reconstruction ...")
recon_mace = mbircone.mace.mace4D(sino_noisy, angles, dist_source_detector, magnification,
                                  denoiser=denoiser, denoiser_args=(),
                                  max_admm_itr=max_admm_itr, prior_weight=prior_weight,
                                  cluster=cluster, min_nb_start_worker=maximum_possible_nb_worker//2,
                                  delta_pixel_detector=delta_pixel_detector,
                                  weight_type='transmission')
recon_shape = recon_mace.shape
print("Reconstruction shape = ", recon_shape)


# ###########################################################################
# Post-process reconstruction results
# ###########################################################################
print("Post processing MACE reconstruction results ...")
# Save recon results as a numpy array
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "recon_mace.npy"), recon_mace)

# Plot axial slices of phantom and recon
display_slices = [2, 4, 6]
for t in range(5):
    for display_slice in display_slices:
        demo_utils.plot_image(phantom[t,display_slice,:,:], title=f'phantom, axial slice {display_slice}, time point {t}',
                              filename=os.path.join(output_dir, f'phantom_slice{display_slice}_time{t}.png'), vmin=0, vmax=0.5)
        demo_utils.plot_image(recon_mace[t,display_slice,:,:], title=f'MACE reconstruction, axial slice {display_slice}, time point {t}',
                              filename=os.path.join(output_dir, f'recon_mace_slice{display_slice}_time{t}.png'), vmin=0, vmax=0.5)

    # Plot 3D phantom and recon image volumes as gif images.
    demo_utils.plot_gif(phantom_orig[t], output_dir, f'phantom_original_{t}', vmin=0, vmax=0.5)
    demo_utils.plot_gif(phantom[t], output_dir, f'phantom_resized_{t}', vmin=0, vmax=0.5)
    demo_utils.plot_gif(recon_mace[t], output_dir, f'recon_mace_{t}', vmin=0, vmax=0.5)

input("press Enter")
