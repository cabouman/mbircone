import numpy as np
import os
import mbircone
import argparse
import getpass
from psutil import cpu_count
from demo_utils import load_yaml, plt_cmp_3dobj
from scipy import ndimage

if __name__ == '__main__':
    # Ask for configuration file
    parser = argparse.ArgumentParser(description='Get configs path')
    parser.add_argument('--configs_path', type=str, default=None, help="Configs path")
    args = parser.parse_args()


    num_parallel = 8

    # Set sinogram shape
    num_det_rows = 200
    num_det_channels = 128
    num_views = 144

    # Reconstruction parameters
    sharpness = 0.2
    snr_db = 31.0

    # magnification is unitless.
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


    # Obtain Cluster or use can define cluster based on below webpage.
    # API of dask_jobqueue https://jobqueue.dask.org/en/latest/api.html
    # API of https://docs.dask.org/en/latest/setup/single-distributed.html#localcluster
    if args.configs_path is None:
        num_cpus = cpu_count(logical=False)
        num_worker_per_node = int(np.sqrt(num_cpus))
        cluster, min_nb_worker = mbircone.parallel_utils.get_cluster_ticket(
            'LocalHost',
            num_worker_per_node=num_worker_per_node)
        num_threads = num_cpus // num_worker_per_node

    else:
        # Load cluster setup parameter.
        configs = load_yaml(args.configs_path)
        # Set openmp number of threads
        num_threads = configs['cluster_params']['num_threads_per_worker']

        if configs['job_queue_sys'] == 'LocalHost':
            cluster, min_nb_worker = mbircone.parallel_utils.get_cluster_ticket(
                configs['job_queue_sys'],
                num_worker_per_node=configs['cluster_params']['num_worker_per_node'])
        else:
            cluster, min_nb_worker = mbircone.parallel_utils.get_cluster_ticket(
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

    # Generate a phantom
    phantom = mbircone.phantom.gen_shepp_logan_3d(num_rows_cols, num_rows_cols, num_slices_phantom)
    print('Generated phantom shape = ', np.shape(phantom))
    phantom = mbircone.cone3D.pad_roi2ror(phantom, boundary_size)
    print('Padded phantom shape = ', np.shape(phantom))

    # Generate simulated data using forward projector on the 3D shepp logan phantom.
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    phantom_rot_para = np.linspace(0, 180, num_parallel, endpoint=False)
    #phantom_list = [ndimage.rotate(phantom, phantom_rot_ang, order=0, mode='constant', axes=(1, 2), reshape=False) for phantom_rot_ang in phantom_rot_para]
    variable_args_list = [{'angle': phantom_rot_ang} for phantom_rot_ang in phantom_rot_para]
    fixed_args = {'input': phantom,
                'order': 0,
                'mode': 'constant',
                'axes':(1, 2),
                'reshape': False}
    phantom_list = mbircone.parallel_utils.scatter_gather(ndimage.rotate,
                                                        variable_args_list=variable_args_list,
                                                        fixed_args=fixed_args,
                                                        cluster=cluster,
                                                        min_nb_worker=min_nb_worker,
                                                        verbose=1)

    # After setting the geometric parameter, the shape of the input phantom should be equal to the calculated geometric parameter.
    # Input a phantom with wrong shape will generate a bunch of issue in C.
    sino_list = [mbircone.cone3D.project(phantom_rot, angles,
                                   num_det_rows=num_det_rows, num_det_channels=num_det_channels,
                                   dist_source_detector=dist_source_detector, magnification=magnification,
                                   delta_pixel_detector=delta_pixel_detector, delta_pixel_image=delta_pixel_image,
                                   channel_offset=channel_offset, row_offset=row_offset) for phantom_rot in phantom_list]
    angles_list = [np.copy(angles) for i in range(num_parallel)]




    # Stacked 3D reconstruction
    variable_args_list = [{'sino': sino, 'angles': angles} for sino, angles in zip(sino_list, angles_list)]
    fixed_args = {'dist_source_detector': dist_source_detector,
                'magnification': magnification,
                'delta_pixel_detector': delta_pixel_detector,
                'delta_pixel_image':delta_pixel_image,
                'channel_offset': channel_offset,
                'row_offset': row_offset,
                'sharpness': sharpness,
                'snr_db':snr_db,
                'max_iterations': max_iterations,
                'num_threads': num_threads,
                'verbose': 1}
    recon_list = mbircone.parallel_utils.scatter_gather(mbircone.cone3D.recon,
                                                        variable_args_list=variable_args_list,
                                                        fixed_args=fixed_args,
                                                        cluster=cluster,
                                                        min_nb_worker=min_nb_worker,
                                                        verbose=1)

    print(np.array(recon_list).shape)


    # create output folder and save reconstruction list
    os.makedirs('output/3D_shepp_logan/', exist_ok=True)
    np.save("./output/3D_shepp_logan/recon_sh4d.npy", np.array(recon_list))


    #Display and compare reconstruction
    for i in range(num_parallel):
        plt_cmp_3dobj(phantom_list[i], recon_list[i], display_slice, display_x, display_y, vmin, vmax, filename % i)
    input("press Enter")