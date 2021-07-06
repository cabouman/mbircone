
import os
import numpy as np
import mbircone
import demo_utils
import pprint


dataset_dir = "/depot/bouman/users/li3120/datasets/metal_weld_data/"
path_radiographs = dataset_dir+"Radiographs-2102414-2019-001-076/"
num_views = 100
downsample_factor=[8, 8]
crop_factor=[[0.25,0.05],[0.75,0.95]]

NSI_system_params = mbircone.preprocess.read_NSI_params("/depot/bouman/users/li3120/datasets/metal_weld_data/2102414-2019-001-076_29.58um_0.5BH.nsipro")

sino, angles= mbircone.preprocess.preprocess(path_radiographs, 
    path_blank=dataset_dir + 'Corrections/gain0.tif', 
    path_dark=dataset_dir + 'Corrections/offset.tif',
    view_range=[0, NSI_system_params["num_acquired_scans"]-1], 
    total_angles=NSI_system_params["total_angles"], 
    num_views=num_views, 
    num_acquired_scans=NSI_system_params["num_acquired_scans"],
    rotation_direction="negative", 
    downsample_factor=downsample_factor, 
    crop_factor=crop_factor)

print("sinogram shape:", sino.shape)
print("Angles List:")
print(angles)

pp=pprint.PrettyPrinter(indent=4)

NSI_system_params=mbircone.preprocess.adjust_NSI_sysparam(NSI_system_params,
    downsample_factor=downsample_factor, 
    crop_factor=crop_factor)

geo_params = mbircone.preprocess.transfer_NSI_to_MBIRCONE(NSI_system_params)
print("NSI system paramemters:")
pp.pprint(NSI_system_params)
print("MBIRCONE Geometric paramemters:")
pp.pprint(geo_params)

dist_source_detector = geo_params["dist_source_detector"]
magnification = geo_params["magnification"]
delta_pixel_detector = geo_params["delta_pixel_detector"]
channel_offset = geo_params["channel_offset"]-0.1
row_offset = geo_params["row_offset"]



x = mbircone.cone3D.recon(sino, angles, dist_source_detector=dist_source_detector, magnification=magnification, 
	delta_pixel_detector=delta_pixel_detector,
	channel_offset=channel_offset, row_offset=row_offset, weight_type='transmission', p=1.15, q=2.2, sharpness=1, num_neighbors=26,
	max_iterations=20,lib_path='./output',num_threads=4)

# create output folder
os.makedirs('output', exist_ok=True)

np.save("./output/recon.npy",x)

demo_utils.plot_image(sino[0], title='sino', filename='output/sino_%d.png'%0)
slice_index = 20
demo_utils.plot_image(x[slice_index], title='recon', filename='output/recon_%d.png'%slice_index,vmin=0,vmax=0.03)
slice_index = 30
demo_utils.plot_image(x[slice_index], title='recon', filename='output/recon_%d.png'%slice_index,vmin=0,vmax=0.03)
slice_index = 40
demo_utils.plot_image(x[slice_index], title='recon', filename='output/recon_%d.png'%slice_index,vmin=0,vmax=0.03)
slice_index = 48
demo_utils.plot_image(x[slice_index], title='recon', filename='output/recon_%d.png'%slice_index,vmin=0,vmax=0.03)
