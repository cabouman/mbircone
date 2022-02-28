import os
import time
import sys
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import pdb

from utils_test import *
from utils_train import *


def dncnn(is_training=True, size_z_in=1, size_z_out=1, numLayers=17, width=64):
    network_in = Input(shape=(None,None,size_z_in))
    layer_out = Conv2D(filters=width, kernel_size=(3,3), kernel_initializer='Orthogonal', padding='same', activation='relu')(network_in)
    for layers in range(2, numLayers):
        layer_out = Conv2D(filters=width, kernel_size=(3,3), kernel_initializer='Orthogonal', padding='same', use_bias=False)(layer_out)
        layer_out = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001)(layer_out) 
        layer_out = Activation('relu')(layer_out) 
    layer_out = Conv2D(filters=size_z_out, kernel_size=(3,3), kernel_initializer='Orthogonal', padding='same', use_bias=False)(layer_out)
    network_in_sliced = semi2DCNN_select_z_out_from_z_in(network_in, size_z_in, size_z_out)
    network_out = Subtract()([network_in_sliced, layer_out])
    model = Model(inputs=network_in, outputs=network_out)    

    return model


class denoiser:
    def __init__(self, size_z_in, size_z_out, numLayers, width, learning_rate, is_training=True):
        self.size_z_in = size_z_in
        self.size_z_out = size_z_out
        self.numLayers = numLayers
        self.width = width
        self.is_training = is_training
        # build model
        self.model = dncnn(is_training=self.is_training,
                           size_z_in=self.size_z_in, size_z_out=self.size_z_out, 
                           numLayers=self.numLayers, width=self.width)

        self.model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate, clipvalue=1.0))

    def train(self, clean_patches_train, params):              
        clean_data, noisy_data = generateCleanNoisyPair_All(clean_patches_train, params) 
        history = self.model.fit(noisy_data, clean_data, epochs=params['epoch_count'], batch_size=params['batch_size'],shuffle=True)
        model_json = self.model.to_json()
        with open(os.path.join(params['paths']['out_dir'], "model.json"), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(os.path.join(params['paths']['out_dir'], "model.h5"))
        print("model saved to disk")


