import os, sys
sys.path.append("/scratch/gilbreth/yang1467/lilly_exp/diyu/experiments/")
import numpy as np
from mbircone import preprocess
import matplotlib.pyplot as plt
from utils import calc_upper_range
from utils_train import addNoise_batch
from glob import glob
from exp_utils import plot_image

output_dir = f"output/calc_upper_range_test/"
os.makedirs(output_dir, exist_ok=True)

train_path = '/depot/bouman/data/share_wenrui_diyu/training/16955-2014-1-057-AT 18 inlb Y Slices'
train_path_list = sorted(glob(os.path.join(train_path, '*')))
recon = np.array([preprocess._read_scan_img(img_path) for img_path in train_path_list])
upper_range, recon_smooth, indicator = calc_upper_range(recon)
print("upper_range from clean data = ", upper_range)
recon_noisy = addNoise_batch(recon, 0.1, upper_range)
upper_range_noisy, recon_smooth_noisy, indicator_noisy = calc_upper_range(recon_noisy)
print("upper_range from noisy data = ", upper_range_noisy)

print("\n*******************************************************",
      "\n************* Generating MACE recon plots *************",
      "\n*******************************************************")
vmin=-0.1
vmax=1.1
num_slices = recon.shape[0]
for display_slice in range(0, num_slices):
    plot_image(recon[display_slice,:,:], title=f"Ground truth, axial slice {display_slice}", filename=os.path.join(output_dir, f"train_gd_slice{display_slice}.png"), vmin=vmin, vmax=vmax)
    plot_image(recon_smooth[display_slice,:,:], title=f"smoothed ground truth, axial slice {display_slice}", filename=os.path.join(output_dir, f"train_gd_smoothed_slice{display_slice}.png"), vmin=vmin, vmax=vmax)
    plot_image(indicator[display_slice,:,:], title=f"indicator of ground truth, axial slice {display_slice}", filename=os.path.join(output_dir, f"indicator_gd_slice{display_slice}.png"), vmin=vmin, vmax=vmax)
    plot_image(recon_noisy[display_slice,:,:], title=f"Noisy, axial slice {display_slice}", filename=os.path.join(output_dir, f"train_noisy_slice{display_slice}.png"), vmin=vmin, vmax=vmax)
    plot_image(recon_smooth_noisy[display_slice,:,:], title=f"smoothed noisy, axial slice {display_slice}", filename=os.path.join(output_dir, f"train_noisy_smoothed_slice{display_slice}.png"), vmin=vmin, vmax=vmax)
    plot_image(indicator_noisy[display_slice,:,:], title=f"indicator of noisy, axial slice {display_slice}", filename=os.path.join(output_dir, f"indicator_noisy_slice{display_slice}.png"), vmin=vmin, vmax=vmax)
