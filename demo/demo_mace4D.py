import os, sys
import numpy as np
import argparse
import urllib.request
import tarfile
from keras.models import model_from_json
import getpass
from psutil import cpu_count
from scipy import ndimage
import mbircone
import demo_utils, denoiser_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__=='__main__':
    """
    This script is a demonstration of the mace4D reconstruction algorithm.  Demo functionality includes
     * downloading 3D phantom and denoiser data from specified urls
     * generating 4D simulated data by rotating the 3D phantom with positive angular steps with each time point ... 
     * obtaining a cluster ticket with get_cluster_ticket().
     * Performing multinode computation with the cluster ticket. This includes:
        * generating sinogram data by projecting each phantom at each timepoint, and then adding transmission noise
        * performing a 4D MACE reconstruction.
    """
    print('This script is a demonstration of the mace4D reconstruction algorithm.  Demo functionality includes \
    \n\t * downloading 3D phantom and denoiser data from specified urls. \
    \n\t * generating 4D simulated data by rotating the 3D phantom with positive angular steps with each time point. \
    \n\t * obtaining a cluster ticket with get_cluster_ticket(). \
    \n\t * Performing multinode computation with the cluster ticket. This includes: \
    \n\t    * generating sinogram data by projecting each phantom at each timepoint, and then adding transmission noise \
    \n\t    * performing a 4D MACE reconstruction.')

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
    phantom_name = 'bottle_cap_3D'
    denoiser_name = denoiser_type

    # destination path to download and extract the phantom and NN weight files.
    target_dir = './demo_data/'   
    # path to store output recon images
    output_dir = './output/mace4D_fast/'  

    # Geometry parameters
    dist_source_detector = 839.0472     # Distance between the X-ray source and the detector in units of ALU
    magnification = 5.572128439964856   # magnification = (source to detector distance)/(source to center-of-rotation distance)
    delta_pixel_detector = 0.25         # Scalar value of detector pixel spacing in units of ALU
    num_det_rows = 28                   # number of detector rows
    num_det_channels = 240              # number of detector channels

    # Simulated 4D phantom and sinogram parameters
    num_time_points = 4 # number of time points. This is also number of jobs that can be parallellized with multinode computation.
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
    # Load the parameters from configuration file to obtain a cluster ticket.
    # ###########################################################################

    # Ask for a configuration file to obtain a cluster ticket to access a parallel cluster.
    # If the configuration file is not provided, it will automatically set up a LocalCluster based on the number of
    # cores in the local computer and return a ticket needed for :py:func:`~multinode.scatter_gather`.
    parser = argparse.ArgumentParser(description='A demo help users use mbircone.multinode module on High Performance Computer.')
    parser.add_argument('--configs_path', type=str, default=None, help="Path to a configuration file.")
    args = parser.parse_args()
    save_config_dir = './configs/multinode/'
    

    # ###########################################################################
    # Get cluster object for multi-node computation
    # ###########################################################################
    is_multinode = demo_utils.query_yes_no("Are you on a multinode cluster?", default="y")
    if not is_multinode:
        print("Running computation on a single node machine. There is no multinode parallelization in this case.")
        num_physical_cores = cpu_count(logical=False)
        cluster_ticket = mbircone.multinode.get_cluster_ticket('LocalHost', num_physical_cores_per_node=num_physical_cores)
    else:
        print("Running computation on a multinode cluster.")
        if args.configs_path is None:
            print("Config file path not provided. Please provide the correct answers to the following questions, so that we could set up the multinode computation:")
            # create output folder
            os.makedirs(save_config_dir, exist_ok=True)
            # Create a configuration dictionary for a cluster ticket, by collecting required information from terminal.
            configs = demo_utils.create_cluster_ticket_configs(save_config_dir=save_config_dir)
            save_config_path = os.path.join(save_config_dir, 'default.yaml')
            input(f"Cluster config file saved at {save_config_path}. Press Enter to continue:")
        else:
            # Load cluster setup parameter.
            print(f"Loading cluster information from {args.configs_path}")
            configs = demo_utils.load_yaml(args.configs_path)

        cluster_ticket = mbircone.multinode.get_cluster_ticket(
                        job_queue_system_type=configs['job_queue_system_type'],
            num_physical_cores_per_node=configs['cluster_params']['num_physical_cores_per_node'],
            num_nodes=configs['cluster_params']['num_nodes'],
            maximum_memory_per_node=configs['cluster_params']['maximum_memory_per_node'],
            maximum_allowable_walltime=configs['cluster_params']['maximum_allowable_walltime'],
            system_specific_args=configs['cluster_params']['system_specific_args'],
            local_directory=configs['cluster_params']['local_directory'],
            log_directory=configs['cluster_params']['log_directory'])

        print(cluster_ticket)


    # ###########################################################################
    # Generate downsampled phantom 
    # ###########################################################################
    print("Generating downsampled 3D phantom volume ...")

    # load original phantom
    phantom_3D = np.load(phantom_path)
    print("shape of 3D phantom = ", phantom_3D.shape)

    # ###########################################################################
    # Generate a 4D phantom.
    # ########################################################################### 
    print("Generating 4D simulated data by rotating the 3D shepp logan phantom with positive angular steps with each time point ...")
    
    # Create the rotation angles and argument lists, and distribute to workers.
    phantom_rot_para = np.linspace(0, 180, num_time_points, endpoint=False)  # Phantom rotation angles.
    phantom_4D = np.array([ndimage.rotate(input=phantom_3D,
                                   angle=phantom_rot_ang,
                                   order=0,
                                   mode='constant',
                                   axes=(1, 2),
                                   reshape=False) for phantom_rot_ang in phantom_rot_para]) 

    print("shape of 4D phantom = ", phantom_4D.shape)
    # ###########################################################################
    # Generate sinogram
    # ###########################################################################

    print("****Multinode computation with Dask****: Generating sinogram data by projecting each phantom at each timepoint ...")
    # scatter_gather parallel computes mbircone.cone3D.project
    # Generate sinogram data by projecting each phantom in phantom list.
    # Create the projection angles and argument lists, and distribute to workers.
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)  # Same for all time points.
    # After setting the geometric parameter, the shape of the input phantom should be equal to the calculated
    # geometric parameter. Input a phantom with wrong shape will generate a bunch of issue in C.
    variable_args_list = [{'image': phantom_rot} for phantom_rot in phantom_4D]
    constant_args = {'angles': angles,
                     'num_det_rows': num_det_rows,
                     'num_det_channels': num_det_channels,
                     'dist_source_detector': dist_source_detector,
                     'magnification': magnification,
                     'delta_pixel_detector': delta_pixel_detector}
    sino_list = mbircone.multinode.scatter_gather(cluster_ticket,
                                                  mbircone.cone3D.project,
                                                  constant_args=constant_args,
                                                  variable_args_list=variable_args_list, verbose=1)
    sino = np.array(sino_list)
    weights = mbircone.cone3D.calc_weights(sino, weight_type='transmission')

    # Add transmission noise
    noise = sino_noise_sigma * 1. / np.sqrt(weights) * np.random.normal(size=sino.shape)
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
    print("****Multinode computation with Dask****: Performing MACE reconstruction ...")
    recon_mace = mbircone.mace.mace4D(sino_noisy, angles, dist_source_detector, magnification,
                                      denoiser=denoiser, denoiser_args=(),
                                      max_admm_itr=max_admm_itr, prior_weight=prior_weight,
                                      cluster_ticket=cluster_ticket,                                      
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
    display_slices = [4, 7, 10]
    for t in range(num_time_points//4,3*num_time_points//4):
        for display_slice in display_slices:
            demo_utils.plot_image(phantom_4D[t,display_slice,:,:], 
                                  title=f"phantom, axial slice {display_slice}, time point {t}", 
                                  filename=os.path.join(output_dir, f"phantom_slice{display_slice}_time{t}.png"), 
                                  vmin=0, vmax=0.5)
            demo_utils.plot_image(recon_mace[t,display_slice,:,:], title=f"MACE reconstruction, axial slice {display_slice}, time point {t}",
                                  filename=os.path.join(output_dir, f"recon_mace_slice{display_slice}_time{t}.png"), vmin=0, vmax=0.5)

        # Plot 3D phantom and recon image volumes as gif images.
        demo_utils.plot_gif(phantom_4D[t], output_dir, f'phantom_resized_{t}', vmin=0, vmax=0.5)
        demo_utils.plot_gif(recon_mace[t], output_dir, f'recon_mace_{t}', vmin=0, vmax=0.5)

    input("press Enter")
    print(f"Reconstruction results saved in {output_dir}")
