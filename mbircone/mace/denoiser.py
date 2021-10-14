import numpy as np
import os

def _from_tensor(img):
    """ Given a output tensor from keras model, convert it back to a numpy array by removing axes of length one.
        This function is originally written by Kai Zhang in his DnCN code: https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_keras/main_test.py.
    """
    return np.squeeze(np.moveaxis(img[...,0],0,-1))


def _to_tensor(img):
    """ Given a numpy array, convert it to tensor shape that is accepted by keras model
        This function is originally written by Kai Zhang in his DnCN
 code: https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_keras/main_test.py.
    """
    if img.ndim == 2:
        return img[np.newaxis,...,np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img,2,0)[...,np.newaxis]


def keras_denoiser(img_noisy, denoiser_model):
    """ This is an example of a keras denoiser function. This denoiser works with either 3D or 4D image batch.
        Args:
            img_noisy (ndarray): noisy image batch with shape (N_batch, N0, N1, ... , Nm).
            denoiser_model (image_dimeras model instance): A pre-trained keras denoiser model.
        Returns:
            ndarray: denoised image batch with shape (N_batch, N0, N1, ... , Nm).
    """
    tensor_noisy = _to_tensor(img_noisy)
    tensor_denoised = denoiser_model.predict(tensor_noisy) # inference
    img_denoised = _from_tensor(tensor_denoised)
    return img_denoised

