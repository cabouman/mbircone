import os
import time
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


# helper function to calculate psnr
def tf_psnr(im1, im2, maxval=1.0):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 , predictions=im1 )
    return 10.0 * (tf.log(maxval ** 2 / mse) / tf.log(10.0))


def semi2DCNN_inPad(size_z_in, size_z_out):
    padSize_in1 = (size_z_in-size_z_out)//2
    padSize_in2 = size_z_in-size_z_out-padSize_in1
    return padSize_in1, padSize_in2


def semi2DCNN_select_z_out_from_z_in(img_patch, size_z_in, size_z_out):
    padSize_in1, padSize_in2 = semi2DCNN_inPad(size_z_in, size_z_out)
    return img_patch[:,:,:,padSize_in1:padSize_in1+size_z_out]


class DataLoader:
    def __init__(self, img_list):
        self.size_z_in = 5
        self.size_z_out = 1
        assert len(img_list) != 0, 'img_list empty'
        self.inData = []
        self.outData = []
        self.padSize_in1, self.padSize_in2 = semi2DCNN_inPad(self.size_z_in, self.size_z_out)
        for i,recon in enumerate(img_list):
            assert recon.shape[0]%self.size_z_out == 0, 'shape_z({}) not divisible by size_z_out({})'.format(recon.shape[0],self.size_z_out)
            recon_out = np.copy(recon)
            self.outData.append(recon_out)
            recon_in = np.pad(np.copy(recon), ((self.padSize_in1,self.padSize_in2), (0,0), (0,0)), 'reflect')
            self.inData.append(recon_in)


    def __iter__(self):
        for self.id_t in range(len(self.outData)):
            # iterate in & out data with stride size_z_out starting at 0
            z_out_iter = range(0,self.outData[self.id_t].shape[0],self.size_z_out)
            for (self.id_z_in,self.id_z_out) in zip(z_out_iter,z_out_iter):
                yield self.getInput_current()

    def getInput_current(self):
        return self.getInput(self.id_t, self.id_z_in)


    def setOutput_current(self, img):
        return self.setOutput(self.id_t, self.id_z_out, img)

    
    def getInput(self, id_t, id_z_in):
        img = self.inData[id_t][range(id_z_in,id_z_in+self.size_z_in)] # z x n1 x n2
        img = np.swapaxes(img, 0, 2) # n2 x n1 x z
        img = np.swapaxes(img, 0, 1) # n1 x n2 x z
        img = np.expand_dims(img, axis=0) # 1 x n1 x n2 x z
        img = np.copy(img, order='C') # contiguous order
        return img


    def setOutput(self, id_t, id_z_out, img):
        # shape: 1 x n1 x n2 x z
        img = np.squeeze(img, axis=0) # n1 x n2 x z
        img = np.swapaxes(img, 0, 1) # n2 x n1 x z
        img = np.swapaxes(img, 0, 2) # z x n1 x n2
        self.outData[id_t][range(id_z_out,id_z_out+self.size_z_out)] = img


class DenoiserCT:
    def __init__(self, checkpoint_dir, size_z_in=5, size_z_out=1, numLayers=17, width=64):
        self.checkpoint_dir = checkpoint_dir
        self.sess = tf.Session(config=tf.ConfigProto())
        #self.sess = tf.compat.v1.Session()
        self.size_z_in = size_z_in
        self.size_z_out = size_z_out
        self.numLayers = numLayers
        self.width = width
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # build model
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.size_z_out], name='clean_image')
        self.X  = tf.placeholder(tf.float32, [None, None, None, self.size_z_in], name='noisy_image')
        self.Y = self.dncnn()
        self.R = semi2DCNN_select_z_out_from_z_in(self.X, self.size_z_in, self.size_z_out) - self.Y # residual = input - output
        self.loss = tf.losses.mean_squared_error(labels=self.Y_ , predictions=self.Y )
        self.psnr = tf_psnr(self.Y, self.Y_, 1.0)
        
        optimizer = tf.train.AdamOptimizer(self.learning_rate, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)

        initializer = tf.global_variables_initializer()
        self.sess.run(initializer)
        print("############################## Initialized Model Successfully...")

    
    def dncnn(self):
        with tf.variable_scope('block1'):
            layer_out = tf.layers.conv2d(self.X, self.width, 3, padding='same', activation=tf.nn.relu)

        for layers in range(2, self.numLayers):
            with tf.variable_scope('block%d' % layers):
                layer_out = tf.layers.conv2d(layer_out, self.width, 3, padding='same', name='conv%d' % layers, use_bias=False)
                layer_out = tf.nn.relu(tf.layers.batch_normalization(layer_out, training=self.is_training))
        
        with tf.variable_scope('block17'):
            layer_out = tf.layers.conv2d(layer_out, self.size_z_out, 3, padding='same')

            network_in_sliced = semi2DCNN_select_z_out_from_z_in(self.X, self.size_z_in, self.size_z_out)
            network_out = network_in_sliced - layer_out
        return network_out

    
    def test(self, testData_obj):
        load_model_status, global_step = self.load(self.checkpoint_dir)
        assert load_model_status == True, 'Load weights FAILED from {}'.format(self.checkpoint_dir)
        for idx, noisy_img in enumerate(testData_obj):
            
            denoised_img, _ = self.sess.run([self.Y, self.R], feed_dict={self.X: noisy_img, self.is_training: False})
            testData_obj.setOutput_current(denoised_img)



    def load(self, checkpoint_dir, checkpoint_src_transfer=''):

        if checkpoint_src_transfer:
            # Transfer learning
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
