import os,sys
os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np
from exp_utils import plot_image, plot_gif, load_yaml, nrmse
from mbircone import preprocess
from mace_diag import denoiser_wrapper
import pprint
import matplotlib.pyplot as plt
import time
import denoiser_utils
from PIL import Image
from glob import glob
from utils import calc_upper_range

batch_size = 64
denoiser_path = '/scratch/gilbreth/yang1467/dncnn-video/data/dncnn-4d/stride3/train_new_scaling_sigma0.2_accept_0.2_0.6_allxy_outputs/Checkpoints_all'
denoiser_model = denoiser_utils.DenoiserCT(denoiser_path)
# Load denoiser model structure and weights
def denoiser(img_noisy, image_range_upper=None):
    img_noisy = np.expand_dims(img_noisy, axis=0)
    testData_obj = denoiser_utils.DataLoader(img_noisy)
    img_denoised = denoiser_model.denoise(testData_obj, batch_size=batch_size)
    return np.squeeze(img_denoised)

output_dir = f"output/test_denoiser_train_new_scaling_sigma0.2_accept_0.2_0.6_allxy_stride3/"
os.makedirs(output_dir, exist_ok=True)

#data_dir = f"output/KV4D/downsampled-sino-ds_factor4--ds_view2_tp12/"
#data_dir = f"output/half_viewed/downsampled-sino-ds_factor2--ds_view2/mace4D_12tp_priorweight0.75_macesharpness2.0"
#data_dir = 'output/half_viewed/downsampled-sino-ds_factor2--ds_view2/mace4D_12tp_priorweight0.50_macesharpness1.0_normalization_mode3/'
#recon_noisy = np.load(os.path.join(data_dir, 'W1_itr9.npy'))
train_path = '/depot/bouman/data/share_wenrui_diyu/training/16955-2014-1-057-AT 18 inlb Y Slices'
train_path_list = sorted(glob(os.path.join(train_path, '*')))
recon = np.array([preprocess._read_scan_img(img_path) for img_path in train_path_list])
recon = recon[::2,:,:]
print("recon shape = ", recon.shape)

upper_range_gd = calc_upper_range(recon)
print("upper range calculated from gd = ", upper_range_gd)
sigma_std_dev = 0.2*upper_range_gd
print('std dev of AWGN = ', sigma_std_dev)
recon_noisy = recon + np.random.normal(0,sigma_std_dev,recon.shape)
denoiser_args = ()
start_time = time.time()
recon_denoised = denoiser_wrapper(recon_noisy, denoiser, denoiser_args=denoiser_args, permute_vector=(0,1,2), positivity=True)
print("recon denoised shape = ", recon_denoised.shape)
end_time = time.time()
elapsed_time = end_time-start_time
print(f"Inferencing finished. Elapsed time {elapsed_time:.1f} sec")

print("\n*******************************************************",
      "\n************* Generating MACE recon plots *************",
      "\n*******************************************************")
#vmin=-0.005
#vmax=0.045
vmin=-0.1
vmax=1.1
num_slices = recon_denoised.shape[0]
#tp = 5
for display_slice in range(0, num_slices):
    NRMSE = nrmse(recon_denoised[display_slice,:,:], recon[display_slice,:,:])
    plot_image(recon[display_slice,:,:], title=f"Ground truth, axial slice {display_slice}", filename=os.path.join(output_dir, f"train_gd_slice{display_slice}.png"), vmin=vmin, vmax=vmax)
    plot_image(recon_noisy[display_slice,:,:], title=f"Noisy, axial slice {display_slice}", filename=os.path.join(output_dir, f"train_noisy_slice{display_slice}.png"), vmin=vmin, vmax=vmax)
    plot_image(recon_denoised[display_slice,:,:], title=f"Denoised, axial slice {display_slice}, NRMSE={NRMSE:.5f}", filename=os.path.join(output_dir, f"train_denoised_slice{display_slice}.png"), vmin=vmin, vmax=vmax)
