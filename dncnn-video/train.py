import os
import sys
import argparse
import random
import numpy as np
import pdb
from glob import glob
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import timeit
from ruamel.yaml import YAML
import cnn
from utils_test import *
from utils_train import *
from utils import *
from genpatch import get_patches_from_recon4D, isPatchGood
from mbircone import preprocess

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1)
arg_parser.add_argument('--params_path', dest='params_path', default='./params.yml')

args, extra = arg_parser.parse_known_args()
args = vars(args)


def main():
    params = get_params(args['params_path'])
    print_params(params)
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(params['CUDA_VISIBLE_DEVICES'])
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    os.makedirs(os.path.dirname(params['paths']['checkpoint_dir']), exist_ok=True)
    if params['use_gpu']:
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
    else:
        config = tf.ConfigProto()


    train_path = '/depot/bouman/data/share_conebeam_data/StaticScanExample/Slices/16955-2014-1-057-AT 18 inlb Y Slices'
    train_path_list = sorted(glob(os.path.join(train_path, '*')))
    cleanData_train = np.array([preprocess._read_scan_img(img_path) for img_path in train_path_list])
    np.random.seed(seed=1)
    upper_range = calc_upper_range(cleanData_train, percentile=params['percentile'], gauss_sigma=params['gauss_sigma'], threshold=params['indicator_threshold'])
    print("upper_range from clean data = ", upper_range)
    cleanData_train = [cleanData_train] 
    clean_patches_train_orig = get_patches_from_recon4D(cleanData_train, params)
    print("shape of clean patches before inspection = ", np.shape(clean_patches_train_orig))
    clean_patches_train = []
    for patch_clean in clean_patches_train_orig:
        if isPatchGood(patch_clean, params):
            clean_patches_train.append(patch_clean)

    clean_patches_train = np.stack(clean_patches_train, axis=0) # n x conv1 x conv2 x chan
    np.save(os.path.join(params['paths']['out_dir'], 'clean_patches_raw.npy'), clean_patches_train)
    print("shape of clean patches after inspection = ", np.shape(clean_patches_train))
    permutation = np.random.permutation(clean_patches_train.shape[0])
    clean_patches_train = clean_patches_train[permutation]
    clean_patches_train = clean_patches_train / upper_range
    #np.save(os.path.join(params['paths']['out_dir'], 'clean_patches.npy'), clean_patches_train)
    with tf.Session(config=config) as sess:
        denoiser_obj = cnn.denoiser(sess, size_z_in=params['size_z_in'], size_z_out=params['size_z_out'], numLayers=params['numLayers'], width=params['width'])
        denoiser_obj.train(clean_patches_train, params)



if __name__ == '__main__':
     main()

