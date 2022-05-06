import time
import sys
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from skimage.filters import gaussian

def dncnn(network_in, is_training=True, size_z_in=1, size_z_out=1, numLayers=17, width=64, is_normalize=0):
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


class DenoiserCT:
    def __init__(self, checkpoint_dir, size_z_in=5, size_z_out=1, numLayers=17, width=64):
        self.checkpoint_dir = checkpoint_dir
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)))
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
        self.psnr = tf_psnr(self.Y, self.Y_, 1.0)

        optimizer = tf.train.AdamOptimizer(self.learning_rate, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)

        initializer = tf.global_variables_initializer()
        self.sess.run(initializer)
        print("############################## Initialized Model Successfully...")
        load_model_status = self._load()
        assert load_model_status == True, 'Load weights FAILED from {}'.format(self.checkpoint_dir)
 
    
    def denoise(self, testData_obj, batch_size=None):
        """ Denoise function. This function takes a DataLoader class object as input, denoise the testData_obj.inData, and save the denoised images as testData_obj.outData. 
        """
        noisy_img = np.array([noisy_img[0] for noisy_img in testData_obj])
        n_img = noisy_img.shape[0]
        N_batch, Nt, Nx, Ny = np.shape(testData_obj.inData)
        if batch_size is None:
            denoised_img, _ = self.sess.run([self.Y, self.R], feed_dict={self.X: noisy_img, self.is_training: False})
        else:
            denoised_img = []
            for n in range(0, n_img, batch_size):
                if n+batch_size>n_img:
                    batch_noisy = noisy_img[n:,:,:,:]
                else:
                    batch_noisy = noisy_img[n:n+batch_size,:,:,:]
                denoised_batch, _ = self.sess.run([self.Y, self.R], feed_dict={self.X: batch_noisy, self.is_training: False})
                if not len(denoised_img):
                    denoised_img = np.array(denoised_batch)
                else: 
                    denoised_img = np.append(denoised_img, denoised_batch, axis=0)
        denoised_img = np.reshape(denoised_img, (N_batch, -1, Nx, Ny))
        return denoised_img
    

    def _load(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(self.checkpoint_dir)
            saver.restore(self.sess, full_path)
            load_model_status = True
        else:
            load_model_status = False
        return load_model_status    


class DataLoader:
    """ DataLoader class that will be used in DenoiserCT. 
    """
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


def calc_upper_range(img, percentile=75, gauss_sigma=2., is_segment=True, threshold=0.5):
    '''
    Given a 4D image volume of shape (Nz, Nt, Nx, Ny), calculate the image upper range.
    '''
    assert(np.ndim(img)<=4), 'Error! Input image dim must be 4 or lower!'
    if np.ndim(img) < 4:
        print(f"{np.ndim(img)} dim image provided. Automatically adding axes to make the input a 4D image volume.")
        for _ in range(4-np.ndim(img)):
            img = np.expand_dims(img, axis=0)
    Nz, Nt, _, _ = np.shape(img)
    img_smooth = np.array([[gaussian(img[i,t,:,:], gauss_sigma, preserve_range=True) for t in range(Nt)] for i in range(Nz)])
    if is_segment:
        img_mean = np.mean(img_smooth)
        indicator = img_smooth > threshold * img_mean
    else:
        indicator = np.ones(img.shape)
    img_upper_range = np.percentile(img_smooth[indicator], percentile)
    #return img_upper_range, np.squeeze(img_smooth), np.squeeze(indicator)
    return img_upper_range

