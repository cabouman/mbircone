
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


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1)
arg_parser.add_argument('--params_path', dest='params_path', default='./params.yml')

args, extra = arg_parser.parse_known_args()
args = vars(args)


def partition_training_data(params):

    makeDirInPath(params['paths']['out_dir'])

    cleanData_train = np.load(params['paths']['patches_train_clean'])
    noisyData_train = np.load(params['paths']['patches_train_noisy'])
    assert cleanData_train.shape==noisyData_train.shape, 'Noisy and Clean image have different shape'
    print('Training Data Shape: {}'.format(cleanData_train.shape))

    print('Shuffling Clean and Noisy Training Data')
    permutation = np.random.permutation(cleanData_train.shape[0])
    cleanData_train = cleanData_train[permutation]
    noisyData_train = noisyData_train[permutation]
    print('Shuffle Done')

    total_trainSize = cleanData_train.shape[0]
    evalSize_numBatch = np.int( np.ceil(total_trainSize * params['getEvalFromTrain_fraction'] / params['batch_size']) )
    evalSize = evalSize_numBatch * params['batch_size']
    
    cleanData_eval = cleanData_train[0:evalSize]
    cleanData_train = cleanData_train[evalSize:]
    noisyData_eval = noisyData_train[0:evalSize]
    noisyData_train = noisyData_train[evalSize:]

    print("New Training Data Size: {}".format(cleanData_train.shape[0]))
    print("New Eval Data Size: {}".format(cleanData_eval.shape[0]))

    save_summary(cleanData_train, folderName=os.path.join(params['paths']['out_dir'], 'patch_summary_train_init'), num_patch_summary=100, verbose=0)

    return cleanData_train, cleanData_eval, noisyData_train, noisyData_eval



def main():

    params = get_params(args['params_path'])
    print_params(params)

    #os.environ['CUDA_VISIBLE_DEVICES'] = str(params['CUDA_VISIBLE_DEVICES'])
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    os.makedirs(os.path.dirname(params['paths']['checkpoint_dir']), exist_ok=True)
    
    if params['use_gpu']:
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
    else:
        config = tf.ConfigProto()


    np.random.seed(seed=1)
    cleanData_train, cleanData_eval, noisyData_train, noisyData_eval = partition_training_data(params)

    with tf.Session(config=config) as sess:
        denoiser_obj = cnn.denoiser(sess, size_z_in=params['size_z_in'], size_z_out=params['size_z_out'], numLayers=params['numLayers'], width=params['width'], is_normalize=params['is_normalize'])
        denoiser_obj.evaluate(50, cleanData_eval, noisyData_eval, params)



if __name__ == '__main__':
     main()

