import time
import logging
import socket
import os
# from pprint import pprint
import mbircone
from dask.distributed import Client, as_completed
import numpy as np
from dask_jobqueue import SLURMCluster
import demo_utils


# dask.config.set({"distributed.admin.tick.limit":'3h'})
def mbircone_4d(t):
    dataset_dir = "/depot/bouman/users/li3120/datasets/Kinetic_Vision_4D_CT_Data/"
    path_radiographs = dataset_dir + "Radiographs-210414_Kinetic-Vision_Fixture-Vision_Dynamic-4D_Cap/"
    num_views = 36000
    downsample_factor = [4, 4]
    crop_factor = [(0.05, 0.05), (0.95, 0.95)]
    index_time_points = t

    NSI_system_params = mbircone.preprocess.read_NSI_params(
        "/depot/bouman/users/li3120/datasets/Kinetic_Vision_4D_CT_Data/210414_Kinetic-Vision_Fixture-Vision_Dynamic-4D_Cap.nsipro")

    sino, angles = mbircone.preprocess.preprocess(path_radiographs,
                                                  num_views=num_views,
                                                  path_blank=dataset_dir + 'Corrections/gain0.tif',
                                                  path_dark=dataset_dir + 'Corrections/offset.tif',
                                                  view_range=[0, NSI_system_params["num_acquired_scans"] - 1],
                                                  total_angles=NSI_system_params["total_angles"],
                                                  num_acquired_scans=NSI_system_params["num_acquired_scans"],
                                                  rotation_direction="negative",
                                                  downsample_factor=downsample_factor,
                                                  crop_factor=crop_factor,
                                                  num_time_points=NSI_system_params["total_angles"] // 360,
                                                  time_point=index_time_points)

    # print("sinogram shape:", sino.shape)
    # print("Angles List:")
    # print(angles)

    NSI_system_params = mbircone.preprocess.adjust_NSI_sysparam(NSI_system_params,
                                                                downsample_factor=downsample_factor,
                                                                crop_factor=crop_factor)

    geo_params = mbircone.preprocess.transfer_NSI_to_MBIRCONE(NSI_system_params)
    # print("NSI system paramemters:")
    # pprint(NSI_system_params)
    # print("MBIRCONE Geometric paramemters:")
    # pprint(geo_params)

    dist_source_detector = geo_params["dist_source_detector"]
    magnification = geo_params["magnification"]
    delta_pixel_detector = geo_params["delta_pixel_detector"]
    channel_offset = geo_params["channel_offset"]
    row_offset = geo_params["row_offset"]

    x = mbircone.cone3D.recon(sino, angles, dist_source_detector=dist_source_detector, magnification=magnification,
                              delta_pixel_detector=delta_pixel_detector,
                              channel_offset=channel_offset, row_offset=row_offset, weight_type='transmission', p=1.15,
                              q=2.2, sharpness=1, num_neighbors=26,
                              max_iterations=20, lib_path='./output')

    # create output folder
    # os.makedirs('output/kv4d/', exist_ok=True)

    # fname_ref = '/depot-new/bouman/users/li3120/OpenMBIR-ConeBeam-4D/demo_3/data/conebeam/object.recon'
    # ref = demo_utils.read_ND(fname_ref, 3)
    # ref = np.swapaxes(ref, 0, 2)
    # # ref = np.flip(ref,axis=0)
    # rmse_val = demo_utils.nrmse(x,ref)
    # print("NRMSE between reconstruction and reference: {}".format(rmse_val))

    # np.save("./output/kv4d/recon_t%d.npy"%index_time_points,x)
    # demo_utils.plot_gif(x,'./output/kv4d/','t%d'%index_time_points)
    return {'x': x,
            'time point': t,
            'host': socket.gethostname(),
            'pid': os.getpid(),
            'time': int(time.time())}


def print_dict(result):
    print('{')
    print('time point:', result['time point'])
    print('host:', result['host'])
    print('pid:', result['pid'])
    print('time:', result['time'])
    print('}')


cluster = SLURMCluster(project='standby', processes=2, n_workers=20, walltime='02:00:00', memory='32GB',
                       death_timeout=60, job_extra=['--nodes=1', '--ntasks-per-node=1'],
                       env_extra=['module load anaconda',
                                  'source activate mbircone'],
                       cores=24)

cluster.scale(jobs=4)
print(cluster.job_script())
client = Client(cluster)

nb_workers = 0
while True:
    nb_workers = len(client.scheduler_info()["workers"])
    print('Got {} workers'.format(nb_workers))
    if nb_workers >= 6:
        break
    time.sleep(1)

# futures = client.map(slow_increment, range(0,1000,5))

print('client:', client)

for future in as_completed(client.map(mbircone_4d, range(75, 155, 5))):  # FIX
    result = future.result()
    print_dict(result)
    np.save("./output/kv4d/recon_t%d.npy" % result['time point'], result['x'])
    demo_utils.plot_gif(result['x'], './output/kv4d/', 't%d' % result['time point'])
    # pprint(result)
