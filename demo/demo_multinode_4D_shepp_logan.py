import numpy as np
import os
import mbircone
import argparse
import getpass
from psutil import cpu_count
from demo_utils import load_yaml, plt_cmp_3dobj
from scipy import ndimage

if __name__ == '__main__':
    # Ask for a configuration file to obtain a cluster ticket to access a parallel cluster.
    # If the configuration file is not provided, it will automatically set up a LocalCluster based on the number of
    # cores in the local computer and return a ticket needed for :py:func:`~multinode.scatter_gather`.
    parser = argparse.ArgumentParser(description='Get configs path')
    parser.add_argument('--configs_path', type=str, default=None, help="Configs path")
    args = parser.parse_args()

    # sinogram parameters
    num_det_rows = 200
    num_det_channels = 128
    num_views = 144

    # Reconstruction parameters
    sharpness = 0.2
    snr_db = 31.0

    # magnification is unit-less.
    magnification = 2

    # All distances are in unit of 1 ALU = 1 mm.
    dist_source_detector = 600
    delta_pixel_detector = 0.9
    delta_pixel_image = 1.0
    channel_offset = 0
    row_offset = 0
    max_iterations = 20

    # Display parameters
    vmin = 1.0
    vmax = 1.1
    filename = 'output/3D_shepp_logan/results_%d.png'

    # Parallel computer verbose
    par_verbose = 0

    # Obtain a ticket(LocalCluster, SLURMCluster, SGECluster) needed for :py:func:`~multinode.scatter_gather` to access a parallel cluster.
    # More information of obtaining this ticket can be found in below webpage.
    # API of dask_jobqueue https://jobqueue.dask.org/en/latest/api.html
    # API of https://docs.dask.org/en/latest/setup/single-distributed.html#localcluster
    if args.configs_path is None:
        num_cores = cpu_count(logical=False)
        if num_cores >= 4:
            num_worker_per_node = int(np.sqrt(num_cores))
        else:
            num_worker_per_node = num_cores
        cluster, maximum_possible_nb_worker = mbircone.multinode.get_cluster_ticket('LocalHost',
                                                                                    num_nodes=None,
                                                                                    num_worker_per_node=num_worker_per_node,
                                                                                    num_threads_per_worker=None)
        num_threads = num_cores // num_worker_per_node
        # In this demo, distribute one job to each worker.
        num_parallel = num_worker_per_node

    else:
        # Load cluster setup parameter.
        configs = load_yaml(args.configs_path)
        # Set openmp number of threads
        num_threads = configs['cluster_params']['num_threads_per_worker']

        cluster, maximum_possible_nb_worker = mbircone.multinode.get_cluster_ticket(
            job_queue_system_type=configs['job_queue_system_type'],
            num_nodes=configs['cluster_params']['num_nodes'],
            num_worker_per_node=configs['cluster_params']['num_worker_per_node'],
            num_threads_per_worker=configs['cluster_params']['num_threads_per_worker'],
            maximum_memory_per_node=configs['cluster_params']['maximum_memory_per_node'],
            maximum_allowable_walltime=configs['cluster_params']['maximum_allowable_walltime'],
            system_specific_args=configs['cluster_params']['system_specific_args'],
            local_directory=configs['cluster_params']['local_directory'],
            log_directory=configs['cluster_params']['log_directory'])

        # Automatically set number of parallel jobs to distribute to a parallel cluster.
        if configs['cluster_params']['num_nodes'] is None:
            num_nodes = 1
        else:
            num_nodes = configs['cluster_params']['num_nodes']
        # In this demo, distribute one job to each worker.
        num_parallel = configs['cluster_params']['num_worker_per_node'] * num_nodes

    print(cluster)
    print("Parallel compute 3D conebeam reconstruction on %d timepoints.\n" % num_parallel)

    # Generate a 3D shepp logan phantom.
    ROR, boundary_size = mbircone.cone3D.compute_img_size(num_views, num_det_rows, num_det_channels,
                                                          dist_source_detector,
                                                          magnification,
                                                          channel_offset=channel_offset, row_offset=row_offset,
                                                          delta_pixel_detector=delta_pixel_detector,
                                                          delta_pixel_image=delta_pixel_image)
    Nz, Nx, Ny = ROR
    img_slices_boundary_size, img_rows_boundary_size, img_cols_boundary_size = boundary_size
    print('ROR of the recon is:', (Nz, Nx, Ny))

    # Set phantom parameters to generate a phantom inside ROI according to ROR and boundary_size.
    # All valid pixels should be inside ROI.
    num_rows_cols = Nx - 2 * img_rows_boundary_size  # Assumes a square image
    num_slices_phantom = Nz - 2 * img_slices_boundary_size
    print('ROI and shape of phantom is:', num_slices_phantom, num_rows_cols, num_rows_cols)

    # Set display indexes
    display_slice = img_slices_boundary_size + int(0.4 * num_slices_phantom)
    display_x = num_rows_cols // 2
    display_y = num_rows_cols // 2
    display_view = 0

    # Generate a 3D phantom
    phantom = mbircone.phantom.gen_shepp_logan_3d(num_rows_cols, num_rows_cols, num_slices_phantom)
    print('Generated phantom shape = ', np.shape(phantom))
    phantom = mbircone.cone3D.pad_roi2ror(phantom, boundary_size)
    print('Padded phantom shape = ', np.shape(phantom))

    # Set minimum number of workers to start to maximum possible number of worker.
    min_nb_start_worker = maximum_possible_nb_worker

    # scatter_gather parallel computes ndimage.rotate
    # Generate 4D simulated data by rotating the 3D shepp logan phantom by increasing degree per time point.
    # Create the rotation angles and argument lists, and distribute to workers.
    phantom_rot_para = np.linspace(0, 180, num_parallel, endpoint=False)  # Phantom rotation angles.
    variable_args_list = [{'angle': phantom_rot_ang} for phantom_rot_ang in phantom_rot_para]
    constant_args = {'input': phantom,
                     'order': 0,
                     'mode': 'constant',
                     'axes': (1, 2),
                     'reshape': False}
    phantom_list = mbircone.multinode.scatter_gather(ndimage.rotate, constant_args=constant_args,
                                                     variable_args_list=variable_args_list, cluster_ticket=cluster,
                                                     min_workers=min_nb_start_worker, verbose=par_verbose)
    print("Generate 4D simulated data by rotating the 3D shepp logan phantom by increasing degree per time point. \n")

    # scatter_gather parallel computes mbircone.cone3D.project
    # Generate sinogram data by projecting each phantom in phantom list.
    # Create the projection angles and argument lists, and distribute to workers.
    proj_angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)  # Same for all time points.
    # After setting the geometric parameter, the shape of the input phantom should be equal to the calculated
    # geometric parameter. Input a phantom with wrong shape will generate a bunch of issue in C.
    variable_args_list = [{'image': phantom_rot} for phantom_rot in phantom_list]
    constant_args = {'angles': proj_angles,
                     'num_det_rows': num_det_rows,
                     'num_det_channels': num_det_channels,
                     'dist_source_detector': dist_source_detector,
                     'magnification': magnification,
                     'delta_pixel_detector': delta_pixel_detector,
                     'delta_pixel_image': delta_pixel_image,
                     'channel_offset': channel_offset,
                     'row_offset': row_offset}
    sino_list = mbircone.multinode.scatter_gather(mbircone.cone3D.project, constant_args=constant_args,
                                                  variable_args_list=variable_args_list, cluster_ticket=cluster,
                                                  min_workers=min_nb_start_worker, verbose=par_verbose)
    print("Generate sinogram data by projecting each phantom in all timepoints. \n")

    # scatter_gather parallel computes mbircone.cone3D.recon
    # Reconstruct 3D phantom in all timepoints using mbircone.cone3D.recon.
    # Create the projection angles and argument lists, and distribute to workers.
    angles_list = [np.copy(proj_angles) for i in range(num_parallel)]  # Same for all time points.
    variable_args_list = [{'sino': sino, 'angles': angles} for sino, angles in zip(sino_list, angles_list)]
    constant_args = {'dist_source_detector': dist_source_detector,
                     'magnification': magnification,
                     'delta_pixel_detector': delta_pixel_detector,
                     'delta_pixel_image': delta_pixel_image,
                     'channel_offset': channel_offset,
                     'row_offset': row_offset,
                     'sharpness': sharpness,
                     'snr_db': snr_db,
                     'max_iterations': max_iterations,
                     'num_threads': num_threads,
                     'verbose': 0}
    recon_list = mbircone.multinode.scatter_gather(mbircone.cone3D.recon, constant_args=constant_args,
                                                   variable_args_list=variable_args_list, cluster_ticket=cluster,
                                                   min_workers=min_nb_start_worker, verbose=par_verbose)
    print("Reconstruct 3D phantom in all timepoints. \n")
    print("Reconstructed 4D image shape = ", np.array(recon_list).shape)

    # create output folder and save reconstruction list
    os.makedirs('output/3D_shepp_logan/', exist_ok=True)
    np.save("./output/3D_shepp_logan/recon_sh4d.npy", np.array(recon_list))

    # Display and compare reconstruction
    for i in range(num_parallel):
        plt_cmp_3dobj(phantom_list[i], recon_list[i], display_slice, display_x, display_y, vmin, vmax, filename % i)
    input("press Enter")
