import os
import time
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pdb

from utils import segmentation_and_percentile
from utils_test import *
from utils_train import *

'''
def denoise_4D(img_noisy, checkpoint_dir, permute_vector_4D=[1,2,3,4], inputNormalize_mode='normalize', inputNormalize_limits=[0,1], 
    numLayers=17, width=64, size_z_in=5, size_z_out=1):
    
    print('shape before: {}'.format(img_noisy.shape))
    img_noisy = np.transpose(img_noisy, permute_vector_4D)
    print('shape after: {}'.format(img_noisy.shape))
    img_noisy = normalizeRecon(img_noisy, inputNormalize_mode, inputNormalize_limits)

    img_list = [img_noisy[i] for i in range(img_noisy.shape[0])]
    testData_obj = loader_reconList(img_list, size_z_in=size_z_in, size_z_out=size_z_out)

    with tf.Session(config=tf.ConfigProto()) as sess:
        denoiser_obj = denoiser(sess, size_z_in, size_z_out, numLayers, width)
        denoiser_obj.test(testData_obj, checkpoint_dir)


    img_denoised = np.stack(testData_obj.outData, axis=0)
    img_denoised = np.transpose(img_denoised, compute_inv_permute_vector(permute_vector_4D))
    img_denoised = denormalizeRecon(img_denoised, inputNormalize_mode, inputNormalize_limits)

    return img_denoised
'''

def dncnn(network_in, is_training=True, size_z_in=1, size_z_out=1, numLayers=17, width=64, is_normalize=0, image_range_upper=None):
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
    def __init__(self, sess, size_z_in, size_z_out, numLayers, width, is_normalize=1):

        self.sess = sess
        self.size_z_in = size_z_in
        self.size_z_out = size_z_out
        self.numLayers = numLayers
        self.width = width
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.is_normalize = is_normalize
        # build model
        self.image_range_upper=tf.placeholder(tf.float32, [None], name='image_range_upper')
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.size_z_out], name='clean_image')
        self.X  = tf.placeholder(tf.float32, [None, None, None, self.size_z_in], name='noisy_image')
        self.X_norm = self.X/self.image_range_upper[:,tf.newaxis,tf.newaxis,tf.newaxis]
        self.Y_norm = dncnn(self.X_norm, is_training=self.is_training,
                       size_z_in=self.size_z_in, size_z_out=self.size_z_out, 
                       numLayers=self.numLayers, width=self.width)
        self.Y = self.Y_norm*self.image_range_upper[:,tf.newaxis,tf.newaxis,tf.newaxis]
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


    def train(self, cleanData_train, cleanData_eval, noisyData_train, noisyData_eval, params):              

        learning_rate = params['learning_rate'] * np.ones([params['epoch_count'] ])

        batch_count = get_numBatches(cleanData_train, params['batch_size'])        

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
        
        # make summary
        tf.summary.scalar('train_loss', self.loss)
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.scalar('paramssnr', self.psnr)
        summary_obj_merged = tf.summary.merge_all()
        
        summary_writer_train = tf.summary.FileWriter(os.path.join(params['paths']['out_dir'],'logs','train'), self.sess.graph)
        summary_writer_eval = tf.summary.FileWriter(os.path.join(params['paths']['out_dir'],'logs','eval'))

        
        print("Start training, with start epoch %d start iter %d : " % (epoch_start, iter_num))

        start_time = time.time()
        loss_arr = [[] for _ in range(params['epoch_count'])]
        psnr_arr = [[] for _ in range(params['epoch_count'])]
        for epoch in range(epoch_start, params['epoch_count']):
            for batch_id in range(batch_id_start, batch_count):
            
                clean_batch, noisy_batch = generateCleanNoisyPair(cleanData_train, cleanData_train, batch_id, params)
                if self.is_normalize==1:
                    print("normalize clean and noisy batch!")
                    batch_range_upper = np.percentile(noisy_batch, 95, axis=(1,2,3))
                elif self.is_normalize==2:
                    print("normalize clean and noisy batch with indicator!")
                    batch_range_info = [segmentation_and_percentile(noisy_batch[i], percentile=90) for i in range(noisy_batch.shape[0])]
                    batch_range_upper = np.array([batch_range_info[t][0] for t in range(noisy_batch.shape[0])])
                    indicator = np.array([batch_range_info[t][1] for t in range(noisy_batch.shape[0])])
                    print("shape of batch range upper = ", batch_range_upper.shape)
                else:
                    print("normalization disabled!")
                    batch_range_upper = np.ones((noisy_batch.shape[0],))

                denoised_batch, _, loss, psnr, summary = self.sess.run( [self.Y, self.train_op, self.loss, self.psnr, summary_obj_merged],
                    feed_dict={self.X: noisy_batch, self.Y_: clean_batch,
                               self.learning_rate: learning_rate[epoch], self.is_training: True, 
                               self.image_range_upper: batch_range_upper})
                
                if epoch%10==0 and batch_id%10==0:
                    np.save(f'normalize{self.is_normalize}_clean_batch_id{batch_id}_epoch{epoch}.npy', clean_batch) 
                    np.save(f'normalize{self.is_normalize}_noisy_batch_id{batch_id}_epoch{epoch}.npy', noisy_batch) 

                    np.save(f'normalize{self.is_normalize}_denoised_batch{batch_id}_epoch{epoch}.npy', denoised_batch) 
                    if self.is_normalize == 2:
                        np.save(f'batch_range_upper_batch{batch_id}_epoch{epoch}.npy', batch_range_upper) 
                        np.save(f'indicator_batch{batch_id}_epoch{epoch}.npy', indicator) 
                
                timeTaken = (time.time() - start_time)
                print("Epoch: %2d, batch: %4d/%4d, time: %d s, loss: %.6f, psnr: %.6f" % (epoch, batch_id, batch_count, timeTaken, loss, psnr))
                loss_arr[epoch].append(loss)
                psnr_arr[epoch].append(psnr)

                iter_num += 1
                summary_writer_train.add_summary(summary, global_step=iter_num)

                summary_obj_timeTaken = tf.Summary(value=[tf.Summary.Value(tag="timeTaken", simple_value=timeTaken)])
                summary_obj_epoch = tf.Summary(value=[tf.Summary.Value(tag="epoch", simple_value=epoch)])
                summary_writer_train.add_summary(summary_obj_timeTaken, global_step=iter_num)
                summary_writer_train.add_summary(summary_obj_epoch, global_step=iter_num)
            
            if np.mod(epoch , params['eval_every_epoch']) == 0:

                eval_psnr, eval_loss = self.evaluate(epoch, cleanData_eval, noisyData_eval, params)

                self.save(iter_num, params['paths']['checkpoint_dir'])
                self.save(iter_num, params['paths']['checkpoint_dir']+'_epoch_'+str(epoch))
                save_summary(clean_batch, folderName=os.path.join(params['paths']['out_dir'], 'patch_summary_train', 'epoch_'+str(epoch) ), img_root='clean', num_patch_summary=100, verbose=0)
                save_summary(noisy_batch, folderName=os.path.join(params['paths']['out_dir'], 'patch_summary_train', 'epoch_'+str(epoch) ), img_root='noisy', num_patch_summary=100, verbose=0)
                save_summary(denoised_batch, folderName=os.path.join(params['paths']['out_dir'], 'patch_summary_train', 'epoch_'+str(epoch) ), img_root='denoised', num_patch_summary=100, verbose=0)


                summary_obj_eval_psnr = tf.Summary(value=[tf.Summary.Value(tag="eval_psnr", simple_value=eval_psnr)])
                summary_obj_eval_loss = tf.Summary(value=[tf.Summary.Value(tag="eval_loss", simple_value=eval_loss)])
                summary_writer_eval.add_summary(summary_obj_eval_psnr, global_step=epoch)
                summary_writer_eval.add_summary(summary_obj_eval_loss, global_step=epoch)

            summary_writer_train.flush()
            summary_writer_eval.flush()

        print("############################## Finish training.")
        np.save('normalize{self.is_normalize}_train_loss.npy', loss_arr)
        np.save('normalize{self.is_normalize}_train_psnr.npy', psnr_arr)
        summary_writer_train.close()
        summary_writer_eval.close()



    def evaluate(self, epoch, cleanData_eval, noisyData_eval, params):

        print("############################## Evaluating...")
        
        load_model_status, global_step = self.load(params['paths']['checkpoint_dir'], params['paths']['checkpoint_src_transfer'])
        
        batch_count = get_numBatches(cleanData_eval, params['batch_size'])
        psnr_sum = 0
        mse_sum = 0
        for batch_id in range(batch_count):

            clean_batch, noisy_batch = generateCleanNoisyPair(cleanData_eval, noisyData_eval, batch_id, params)
            if self.is_normalize==1:
                print("normalize clean and noisy batch!")
                batch_range_upper = np.percentile(noisy_batch, 95, axis=(1,2,3))
            elif self.is_normalize==2:
                print("normalize clean and noisy batch with indicator!")
                batch_range_info = [segmentation_and_percentile(noisy_batch[i], percentile=90) for i in range(noisy_batch.shape[0])]
                batch_range_upper = np.array([batch_range_info[t][0] for t in range(noisy_batch.shape[0])])
                indicator = np.array([batch_range_info[t][1] for t in range(noisy_batch.shape[0])])
                print("shape of batch range upper = ", batch_range_upper.shape)
            else:
                print("normalization disabled!")
                batch_range_upper = np.ones((noisy_batch.shape[0],))

            denoised_batch, mse, psnr = self.sess.run( [self.Y, self.loss, self.psnr],
                    feed_dict={self.X: noisy_batch, self.Y_: clean_batch, self.is_training: False,
                               self.image_range_upper: batch_range_upper})

            psnr_sum += psnr * clean_batch.shape[0]
            mse_sum += mse * clean_batch.shape[0]

        save_summary(clean_batch, folderName=os.path.join(params['paths']['out_dir'], 'patch_summary_eval', 'epoch_'+str(epoch) ), img_root='clean', num_patch_summary=100, verbose=0)
        save_summary(noisy_batch, folderName=os.path.join(params['paths']['out_dir'], 'patch_summary_eval', 'epoch_'+str(epoch) ), img_root='noisy', num_patch_summary=100, verbose=0)
        save_summary(denoised_batch, folderName=os.path.join(params['paths']['out_dir'], 'patch_summary_eval', 'epoch_'+str(epoch) ), img_root='denoised', num_patch_summary=100, verbose=0)

        if len(cleanData_eval)==0:
            avg_psnr = 0
            avg_mse = 0
        else:
            avg_psnr = psnr_sum / len(cleanData_eval)
            avg_mse = mse_sum / len(cleanData_eval)
        print("############################## Eval: Average PSNR: %.2f, Average Loss: %.2f ---" % (avg_psnr,avg_mse))

        return avg_psnr, avg_mse


    def test(self, testData_obj, checkpoint_dir):

        print('Loading weights')
        load_model_status, global_step = self.load(checkpoint_dir)
        assert load_model_status == True, 'Load weights FAILED from {}'.format(checkpoint_dir)
        

        print('Denoising images')
        mseIO_all = 0
        start_time = time.time()
        for idx, noisy_img in enumerate(testData_obj):
            if self.is_normalize==1:
                print("normalize clean and noisy batch!")
                batch_range_upper = np.percentile(noisy_img, 95, axis=(1,2,3))
            elif self.is_normalize==2:
                print("normalize clean and noisy batch with indicator!")
                batch_range_info = [segmentation_and_percentile(noisy_batch[i], percentile=90) for i in range(noisy_batch.shape[0])]
                batch_range_upper = np.array([batch_range_info[t][0] for t in range(noisy_batch.shape[0])])
                indicator = np.array([batch_range_info[t][1] for t in range(noisy_batch.shape[0])])
                print("shape of batch range upper = ", batch_range_upper.shape)
            else:
                print("normalization disabled!")
                batch_range_upper = np.ones((noisy_batch.shape[0],))

            denoised_img, diff_img = self.sess.run([self.Y, self.R], feed_dict={self.X: noisy_img, self.is_training: False, self.image_range_upper: batch_range_upper})
            mseIO_all += np.mean(diff_img ** 2)
            
            testData_obj.setOutput_current(denoised_img)

                # print("Batch: %d" % (idx))
                # sys.stdout.flush()     

        print("Number of 2.5D images: {}".format(idx+1))
        rmseIO_all = np.sqrt( mseIO_all / (idx+1) )
        elepsed_time = time.time() - start_time
        print("--- RMSE(test dataset) %.3f ---" % rmseIO_all)
        print("--- Elepsed Time %.3f ---" % elepsed_time)



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

