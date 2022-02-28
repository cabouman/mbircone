import os
import sys
import argparse
import random
import numpy as np
import pdb
from glob import glob
import timeit
from ruamel.yaml import YAML
import cnn_keras
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


def partition_training_data(cleanData_train):


    permutation = np.random.permutation(cleanData_train.shape[0])
    cleanData_train = cleanData_train[permutation]
    print('Shuffle Done')
    return cleanData_train, cleanData_eval, noisyData_train, noisyData_eval



def main():
    params = get_params(args['params_path'])
    print_params(params)
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(params['CUDA_VISIBLE_DEVICES'])
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    os.makedirs(os.path.dirname(params['paths']['checkpoint_dir']), exist_ok=True)

    train_path = '/depot/bouman/data/share_wenrui_diyu/training/16955-2014-1-057-AT 18 inlb Y Slices'
    train_path_list = sorted(glob(os.path.join(train_path, '*')))
    cleanData_train = np.array([preprocess._read_scan_img(img_path) for img_path in train_path_list])
    np.random.seed(seed=1)
    #upper_range = calc_upper_range(cleanData_train, percentile=params['percentile'], gauss_sigma=params['gauss_sigma'], threshold=params['indicator_threshold'])
    #print("upper_range from clean data = ", upper_range)
    cleanData_train = [cleanData_train] 
    clean_patches_train_orig = get_patches_from_recon4D(cleanData_train, params)
    print("shape of clean patches before inspection = ", np.shape(clean_patches_train_orig))
    clean_patches_train = []
    for patch_clean in clean_patches_train_orig:
        if isPatchGood(patch_clean, params):
            clean_patches_train.append(patch_clean)

    clean_patches_train = np.stack(clean_patches_train, axis=0) # n x conv1 x conv2 x chan
    print("shape of clean patches after inspection = ", np.shape(clean_patches_train))
    permutation = np.random.permutation(clean_patches_train.shape[0])
    clean_patches_train = clean_patches_train[permutation]
    #clean_patches_train = clean_patches_train / upper_range
    np.save(os.path.join(params['paths']['out_dir'], 'clean_patches.npy'), clean_patches_train)
    denoiser_obj = cnn_keras.denoiser(size_z_in=params['size_z_in'], size_z_out=params['size_z_out'], numLayers=params['numLayers'], width=params['width'], learning_rate=params['learning_rate'], is_training=True)
    denoiser_obj.train(clean_patches_train, params)



if __name__ == '__main__':
     main()

