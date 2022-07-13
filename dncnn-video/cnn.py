import os
import time
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pdb

from utils_test import *
from utils_train import *


def dncnn(network_in, is_training=True, size_z_in=1, size_z_out=1, numLayers=17, width=64):
    with tf.variable_scope('block1'):
        layer_out = tf.layers.conv2d(network_in, width, 3, padding='same', activation=tf.nn.relu)

    for layers in range(2, numLayers):
        with tf.variable_scope('block%d' % layers):
            layer_out = tf.layers.conv2d(layer_out, width, 3, padding='same', name='conv%d' % layers, use_bias=False)
            # print(layer_out.shape)
            layer_out = tf.nn.relu(tf.layers.batch_normalization(layer_out, training=is_training))
    
    with tf.variable_scope('block17'):
        layer_out = tf.layers.conv2d(layer_out, size_z_out, 3, padding='same')

        # network_out = network_in - layer_out
        network_in_sliced = semi2DCNN_select_z_out_from_z_in(network_in, size_z_in, size_z_out)
        network_out = network_in_sliced - layer_out
        # print("denoised shape in dncnn: {}".format(network_out.shape))

    return network_out


def tf_psnr(im1, im2, maxval=1.0):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 , predictions=im1 )
    return 10.0 * (tf.log(maxval ** 2 / mse) / tf.log(10.0))


class denoiser:
    def __init__(self, sess, size_z_in, size_z_out, numLayers, width):

        self.sess = sess
        self.size_z_in = size_z_in
        self.size_z_out = size_z_out
        self.numLayers = numLayers
        self.width = width
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # build model
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.size_z_out], name='clean_image')
        self.X  = tf.placeholder(tf.float32, [None, None, None, self.size_z_in], name='noisy_image')
        self.Y = dncnn(self.X, is_training=self.is_training,
                       size_z_in=self.size_z_in, size_z_out=self.size_z_out, 
                       numLayers=self.numLayers, width=self.width)
        self.R = semi2DCNN_select_z_out_from_z_in(self.X, self.size_z_in, self.size_z_out) - self.Y # residual = input - output
        self.loss = tf.losses.mean_squared_error(labels=self.Y_ , predictions=self.Y )
        #self.loss = tf.math.reduce_mean(tf.math.squared_difference(self.Y, self.Y_))
        self.psnr = tf_psnr(self.Y, self.Y_, 1.0)
        
        optimizer = tf.train.AdamOptimizer(self.learning_rate, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)

        initializer = tf.global_variables_initializer()
        self.sess.run(initializer)
        print("############################## Initialized Model Successfully...")


    def train(self, clean_patches_train, params):              
        learning_rate = np.ones([params['epoch_count'] ])
        for i in range(params['epoch_count']//10):
            learning_rate[i*10:(i+1)*10] = params['learning_rate']/(2**i)
        batch_count = get_numBatches(clean_patches_train, params['batch_size'])        
        # load pretrained model
        load_model_status, global_step = self.load(params['paths']['checkpoint_dir'], params['paths']['checkpoint_src_transfer'])
        if load_model_status:
            iter_num = global_step
            epoch_start = global_step // batch_count
            batch_id_start = global_step % batch_count
            print("############################## Model restore success!")
        else:
            iter_num = 0
            epoch_start = 0
            batch_id_start = 0
            print("############################## Did not find pretrained model!")
        
        print("Start training, with start epoch %d start iter %d : " % (epoch_start, iter_num))

        start_time = time.time()
        loss_arr = [[] for _ in range(params['epoch_count'])]
        psnr_arr = [[] for _ in range(params['epoch_count'])]
        print("Clean data shape = ", np.shape(clean_patches_train))
        for epoch in range(epoch_start, params['epoch_count']):
            for batch_id in range(batch_id_start, batch_count):
            
                clean_batch, noisy_batch = generateCleanNoisyPair(clean_patches_train, batch_id, params)
                denoised_batch, residual_batch, _, loss, psnr = self.sess.run( [self.Y, self.R, self.train_op, self.loss, self.psnr],
                    feed_dict={self.X: noisy_batch, self.Y_: clean_batch,
                               self.learning_rate: learning_rate[epoch], self.is_training: True})
                timeTaken = (time.time() - start_time)
                print("Epoch: %2d, batch: %4d/%4d, time: %d s, loss: %.6f, psnr: %.6f" % (epoch, batch_id, batch_count, timeTaken, loss, psnr))
                loss_arr[epoch].append(loss)
                psnr_arr[epoch].append(psnr)
                
                if epoch==params['epoch_count']-1 and batch_id%10==0:
                    np.save(os.path.join(params['paths']['out_dir'], f'clean_batch_id{batch_id}_epoch{epoch}.npy'), clean_batch) 
                    np.save(os.path.join(params['paths']['out_dir'], f'noisy_batch_id{batch_id}_epoch{epoch}.npy'), noisy_batch) 

                    np.save(os.path.join(params['paths']['out_dir'], f'denoised_batch{batch_id}_epoch{epoch}.npy'), denoised_batch) 
                    np.save(os.path.join(params['paths']['out_dir'], f'residual_batch{batch_id}_epoch{epoch}.npy'), residual_batch) 
                iter_num += 1
            
            if np.mod(epoch , params['eval_every_epoch']) == 0:
                self.save(iter_num, params['paths']['checkpoint_dir'])
                self.save(iter_num, params['paths']['checkpoint_dir']+'_epoch_'+str(epoch))

        print("############################## Finish training.")
        np.save(os.path.join(params['paths']['out_dir'], f'train_loss.npy'), loss_arr)
        np.save(os.path.join(params['paths']['out_dir'], f'train_psnr.npy'), psnr_arr)


    def save(self, iter_num, checkpoint_dir, model_name='DnCNN-tensorflow'):
        print("############################## Saving model...")
        
        saver = tf.train.Saver()
        checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=iter_num)


    def load(self, checkpoint_dir, checkpoint_src_transfer=''):
        print("############################## Reading checkpoint...")

        if checkpoint_src_transfer and checkpoint_src_transfer != checkpoint_dir:
            print("Transfer learning!")
            checkpoint_dir = checkpoint_src_transfer

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            load_model_status = True
        else:
            load_model_status = False
            global_step = 0

        if checkpoint_src_transfer:
            # Transfer learning
            global_step = 0

        return load_model_status, global_step

