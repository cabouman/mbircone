import os
import numpy as np
import mbircone
from scipy.ndimage import rotate
import argparse
from psutil import cpu_count
from demo_utils import plot_image, plot_gif, create_cluster_ticket_configs

if __name__ == '__main__':
    """
    This script is a demonstration of how to use scatter_gather() to deploy parallel jobs.  Demo functionality includes
     * obtain cluster ticket with get_cluster_ticket().
     Use scatter_gather to:
     * generate 4D phantom by rotating the 3D shepp logan phantom with positive angular steps with each time point.
     * generate synthetic sinogram data by forward projecting each phantom in all timepoints.
     * perform 3D reconstruction in all timepoints.
    """
    print('This script is a demonstration of how to use scatter_gather() to deploy parallel jobs.  Demo functionality includes \
    \n\t * obtain cluster ticket with get_cluster_ticket(). \
    \n\t Use scatter_gather to:\
    \n\t * generate 4D phantom by rotating the 3D shepp logan phantom with positive angular steps with each time point. \
    \n\t * generate synthetic sinogram data by forward projecting each phantom in all timepoints. \
    \n\t * perform 3D reconstruction in all timepoints.')

    # ###########################################################################
    # Set the parameters to generate the phantom, synthetic sinogram, and do the recon
    # ###########################################################################

    # Change the parameters below for your own use case.
    # Parallel computation parameters
    num_time_points = 4 # number of time points
    par_verbose = 0 # verbosity level for parallel computation
    # Detector size
    num_det_rows = 128
    num_det_channels = 128
    # Geometry parameters
    magnification = 2.0                        # Ratio of (source to detector)/(source to center of rotation)
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

    # Size of phantom
    num_slices_phantom = 128
    num_rows_phantom = 128
    num_cols_phantom = 128
    delta_pixel_phantom = 0.5

    # Size and proportion of phantom within image
    scale=1.0
    offset_x=0.0
    offset_y=0.0
    offset_z=0.0

    # Size of recon
    num_slices_recon = 128
    num_rows_recon = 128
    num_cols_recon = 128
    delta_pixel_recon = delta_pixel_phantom

    # local path to save phantom, sinogram, and reconstruction images
    save_path = f'output/4D_shepp_logan_multinode/'
    os.makedirs(save_path, exist_ok=True)

    print("Getting cluster ticket for parallel computing ...")
    # ###########################################################################
    # Load the parameters from configuration file to obtain a cluster ticket.
    # ###########################################################################

    # Ask for a configuration file to obtain a cluster ticket to access a parallel cluster.
    # If the configuration file is not provided, it will automatically set up a LocalCluster based on the number of
    # cores in the local computer and return a ticket needed for :py:func:`~multinode.scatter_gather`.
    parser = argparse.ArgumentParser(description='A demo help users use mbircone.multinode module on High Performance Computer.')
    parser.add_argument('--configs_path', type=str, default=None, help="Path to a configuration file.")
    parser.add_argument('--no_multinode', default=False, action='store_true', help="Run this demo in a single node machine.")
    args = parser.parse_args()
    save_config_dir = './configs/multinode/'
    
    # ###########################################################################
    # Obtain a cluster ticket.
    # ###########################################################################

    # Obtain a ticket(LocalCluster, SLURMCluster, SGECluster) needed for :py:func:`~multinode.scatter_gather` to access a parallel cluster.
    # More information of obtaining this ticket can be found in below webpage.
    # API of dask_jobqueue https://jobqueue.dask.org/en/latest/api.html
    # API of https://docs.dask.org/en/latest/setup/single-distributed.html#localcluster

    if args.no_multinode:
        num_physical_cores = cpu_count(logical=False)
        cluster_ticket = mbircone.multinode.get_cluster_ticket('LocalHost', num_physical_cores_per_node=num_physical_cores)
    else:
        if args.configs_path is None:
            # create output folder
            os.makedirs(save_config_dir, exist_ok=True)
            # Create a configuration dictionary for a cluster ticket, by collecting required information from terminal.
            configs = create_cluster_ticket_configs(save_config_dir=save_config_dir)

        else:
            # Load cluster setup parameter.
            configs = load_yaml(args.configs_path)

        cluster_ticket = mbircone.multinode.get_cluster_ticket(
            job_queue_system_type=configs['job_queue_system_type'],
            num_physical_cores_per_node=configs['cluster_params']['num_physical_cores_per_node'],
            num_nodes=configs['cluster_params']['num_nodes'],
            maximum_memory_per_node=configs['cluster_params']['maximum_memory_per_node'],
            maximum_allowable_walltime=configs['cluster_params']['maximum_allowable_walltime'],
            system_specific_args=configs['cluster_params']['system_specific_args'],
            local_directory=configs['cluster_params']['local_directory'],
            log_directory=configs['cluster_params']['log_directory'])

    print('Genrating 3D Shepp Logan phantom ...')
    ######################################################################################
    # Generate a 3D shepp logan phantom
    ######################################################################################

    phantom_3D = mbircone.phantom.gen_shepp_logan_3d(num_rows_phantom, num_cols_phantom, num_slices_phantom, scale=scale, offset_x=offset_x, offset_y=offset_y, offset_z=offset_z)
    # scale the phantom by a factor of 10.0 to make the projections physical realistic -log attenuation values
    phantom_3D = phantom_3D/10.0
    print('3D Phantom shape = ', np.shape(phantom_3D))

    # ###########################################################################
    # Generate a 4D shepp logan phantom.
    # ###########################################################################
    print("generate 4D phantom by rotating the 3D phantom with positive angular steps with each time point.")
    # Generate 4D simulated data by rotating the 3D shepp logan phantom by increasing degree per time point.
    # Create the rotation angles and argument lists, and distribute to workers.
    tilt_angles = np.linspace(0, 45, num_time_points, endpoint=False)  # Phantom rotation angles.
    phantom_4D = [rotate(input=phantom_3D,
                         angle=tilt_ang,
                         order=0,
                         mode='constant',
                         axes=(1, 2),
                         reshape=False) for tilt_ang in tilt_angles]
    
    # ###########################################################################
    # Generate synthetic sinogram
    # ###########################################################################

    print("****Multinode computation with Dask****: Generating synthetic sinogram data by projecting each phantom at each timepoint ...")
    # scatter_gather parallel computes mbircone.cone3D.project
    # Generate sinogram data by projecting each phantom in phantom list.
    # Create the projection angles and argument lists, and distribute to workers.
    # After setting the geometric parameter, the shape of the input phantom should be equal to the calculated
    # geometric parameter. Input a phantom with wrong shape will generate a bunch of issue in C.
    variable_args_list = [{'image': phantom_rot} for phantom_rot in phantom_4D]
    constant_args = {'angles': angles,
                     'num_det_rows': num_det_rows,
                     'num_det_channels': num_det_channels,
                     'dist_source_detector': dist_source_detector,
                     'magnification': magnification,
                     'delta_pixel_image': delta_pixel_phantom}
    sino_list = mbircone.multinode.scatter_gather(cluster_ticket,
                                                  mbircone.cone3D.project,
                                                  constant_args=constant_args,
                                                  variable_args_list=variable_args_list,

                                                  verbose=par_verbose)

    # ###########################################################################
    # Perform multinode reconstruction
    # ###########################################################################

    print("****Multinode computation with Dask****: Performing 3D qGGMRF recon at all time points ...")
    # scatter_gather parallel computes mbircone.cone3D.recon
    # Reconstruct 3D phantom in all timepoints using mbircone.cone3D.recon.
    # Create the projection angles and argument lists, and distribute to workers.
    angles_list = [np.copy(angles) for t in range(num_time_points)]  # Same for all time points.
    variable_args_list = [{'sino': sino, 'angles': angles} for sino, angles in zip(sino_list, angles_list)]
    constant_args = {'dist_source_detector': dist_source_detector,
                     'magnification': magnification,
                     'delta_pixel_image': delta_pixel_recon,
                     'num_image_rows': num_rows_recon, 
                     'num_image_cols': num_cols_recon, 
                     'num_image_slices': num_slices_recon, 
                     'sharpness': sharpness, 
                     'T': T}
    recon_list = mbircone.multinode.scatter_gather(cluster_ticket,
                                                   mbircone.cone3D.recon,
                                                   constant_args=constant_args,
                                                   variable_args_list=variable_args_list,
                                                   verbose=par_verbose)

    print('4D recon shape = ', np.shape(recon_list))

    ######################################################################################
    # Generate phantom, synthetic sinogram, and reconstruction images
    ######################################################################################
    # Set display indexes for phantom and recon images
    display_slice_phantom = num_slices_phantom // 2
    display_x_phantom = num_rows_phantom // 2
    display_y_phantom = num_cols_phantom // 2
    display_slice_recon = num_slices_recon // 2
    display_x_recon = num_rows_recon // 2
    display_y_recon = num_cols_recon // 2

    for t in range(num_time_points):
        # phantom images
        plot_image(phantom_4D[t][display_slice_phantom, :, :], title=f'phantom, axial slice {display_slice_phantom}, time point {t}',
                   filename=os.path.join(save_path, f'phantom_axial_tp{t}.png'), vmin=vmin, vmax=vmax)
        plot_image(phantom_4D[t][:,display_x_phantom,:], title=f'phantom, coronal slice {display_x_phantom}, time point {t}',
                   filename=os.path.join(save_path, f'phantom_coronal_tp{t}.png'), vmin=vmin, vmax=vmax)
        plot_image(phantom_4D[t][:,:,display_y_phantom], title=f'phantom, sagittal slice {display_y_phantom}, time point {t}',
                   filename=os.path.join(save_path, f'phantom_sagittal_tp{t}.png'), vmin=vmin, vmax=vmax)
                   
        # recon images
        plot_image(recon_list[t][display_slice_recon, :, :], title=f'qGGMRF recon, axial slice {display_slice_recon}, time point {t}',
                   filename=os.path.join(save_path, f'recon_axial_tp{t}.png'), vmin=vmin, vmax=vmax)
        plot_image(recon_list[t][:,display_x_recon,:], title=f'qGGMRF recon, coronal slice {display_x_recon}, time point {t}',
                   filename=os.path.join(save_path, f'recon_coronal_tp{t}.png'), vmin=vmin, vmax=vmax)
        plot_image(recon_list[t][:,:,display_y_recon], title=f'qGGMRF recon, sagittal slice {display_y_recon}, time point {t}',
                   filename=os.path.join(save_path, f'recon_sagittal_tp{t}.png'), vmin=vmin, vmax=vmax)
                   
    print(f"Images saved to {save_path}.") 
    input("Press Enter")

