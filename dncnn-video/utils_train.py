
import os
from glob import glob
import sys

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image

import array
import struct

import pdb

from utils import *

#############################################################################
## Data Augment
#############################################################################

def generateCleanNoisyPair(clean_data, batch_id, train_params):
    clean_batch = select_batch(clean_data, batch_id, train_params['batch_size'])
    clean_batch = augment_batch(clean_batch, train_params)
    noisy_batch = addNoise_batch(clean_batch, train_params['noise_sigma'])
    clean_batch = semi2DCNN_select_z_out_from_z_in(clean_batch, train_params['size_z_in'], train_params['size_z_out'])

    return clean_batch, noisy_batch

def get_numBatches(clean_data, batch_size, verbose=1):
    # numBatches = np.int( np.ceil(clean_data.shape[0]/batch_size) )
    numBatches =clean_data.shape[0]//batch_size
    if verbose:
        print("Data Size: %d, batch_size: %d, numBatches: %d" %( clean_data.shape[0], batch_size, numBatches ))
    return numBatches

def select_batch(data, batch_id, batch_size):
    id_start = batch_id * batch_size
    id_end = (batch_id + 1) * batch_size
    if (id_end>data.shape[0]-1) :
        id_end=data.shape[0]-1
    batch_view = data[id_start:id_end, :, :, :]
    batch = np.copy(batch_view) # Different memory: original does not change accidentally
    return batch

def augment_batch(clean_batch, train_params):
    batch_size = clean_batch.shape[0]
    for patch_id in range(batch_size):
        clean_batch[patch_id] = augment_patch(clean_batch[patch_id], train_params)
    return clean_batch


def augment_patch(patch, train_params):
    # rotate and flip
    if patch.shape[0]!=1 and patch.shape[1]!=1:
        patch = orient_patch(patch, train_params['is_augOrientOn'])
    if train_params['is_randShift_on']:
        patch = addRandShiftAndGradient_patch(patch, train_params)
    return patch

def orient_patch(patch, is_augOrientOn):
    if is_augOrientOn:
        is_flipud = np.random.randint(2)
        is_fliplr = np.random.randint(2)
        is_flipz = np.random.randint(2)
        is_swapxy = np.random.randint(2)
    else:
        is_flipud = 0
        is_fliplr = 0
        is_flipz = 0
        is_swapxy = 0 
    orig_shape = patch.shape
    if is_flipud==1:
        patch = np.flipud(patch)
    if is_fliplr==1:
        patch = np.fliplr(patch)
    if is_flipz==1:
        patch = np.flip(patch, axis=2)
    if is_swapxy==1:
        patch = np.swapaxes(patch, 0, 1)
    assert patch.shape==orig_shape, 'orient_patch: shape changed'
    return patch


def addRandShiftAndGradient_patch(patch, train_params):
    ''' Shifts 0 to randShift_dark and 1 to randShift_bright
        Applies gradients with range [-randGrad,randGrad] to dark and bright
        Each shifts and radius values are picked randomly and independently.
     '''
    randShift_dark =    np.random.uniform(low=train_params['randShift_dark_lo'],     high=train_params['randShift_dark_hi'])
    randShift_bright =  np.random.uniform(low=train_params['randShift_bright_lo'],   high=train_params['randShift_bright_hi'])
    randGrad_dark =     np.random.uniform(low=train_params['randGrad_dark_lo'],      high=train_params['randGrad_dark_hi'])
    randGrad_bright =   np.random.uniform(low=train_params['randGrad_bright_lo'],    high=train_params['randGrad_bright_hi'])
    sizes = np.array(patch.shape)
    randomGradient_dark = randomGradient(sizes)
    randomGradient_bright = randomGradient(sizes)
    patch = randShift_dark*(1-patch) + randShift_bright*patch \
            + randGrad_dark*(1-patch)*randomGradient_dark + randGrad_bright*patch*randomGradient_bright 
    return patch

def randomGradient(sizes):
    ''' sizes must be 3D. output is strictly [-1,1] '''
    sizes = np.array(sizes)
    range0 = np.linspace(-sizes[0], sizes[0], sizes[0])
    range1 = np.linspace(-sizes[1], sizes[1], sizes[1])
    range2 = np.linspace(-sizes[2], sizes[2], sizes[2])
    v0, v1, v2 = np.meshgrid(range0, range1, range2, indexing='ij')
    n = np.random.multivariate_normal([0], [[1]], 3)
    x = n[0]*v0 + n[1]*v1 + n[2]*v2
    # Scale gradient into range [-1,1]
    x = x - np.min(x)
    m = np.max(x)
    if(m!=0):
        x = x // m
    x = 2*x - 1
    return x

def addNoise_batch(clean_batch, noise_sigma):
    print("Adding AWGN with sigma = ", noise_sigma)
    noise = np.random.normal(scale=noise_sigma, size=clean_batch.shape)
    noisy_batch = clean_batch + noise
    return noisy_batch

def randShift(data, dark, bright):
    # Transforms such that 0 becomes dark and 1 becomes bright
    data = data*(bright-dark) + dark
    return data

