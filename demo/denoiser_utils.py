import numpy as np
import os

def _from_tensor(img):
    """ Given a output tensor from cnn model, convert it back to a numpy array by removing axes of length one.
        This function is originally written by Kai Zhang in his DnCN code: https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_keras/main_test.py.
    """
    return np.squeeze(img)


def _to_tensor(img, data_format):
    """ Given a numpy array, convert it to tensor shape that is accepted by cnn model
        This function is originally written by Kai Zhang in his DnCN
 code: https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_keras/main_test.py.
    """
    if data_format == "channels_last":
        if img.ndim == 2:
            return img[np.newaxis, ..., np.newaxis]
        elif img.ndim == 3:
            return img[..., np.newaxis]
    elif data_format == "channels_first":
        if img.ndim == 2:
            return img[np.newaxis, np.newaxis, ...]
        elif img.ndim == 3:
            return np.expand_dims(img, axis=1)

def cnn_denoiser(img_noisy, denoiser_model, data_format="channels_last"):
    """ This is an example of a cnn denoiser function. This denoiser works with either 3D or 4D image batch.
        Args:
            img_noisy (ndarray): noisy image batch with shape (N_batch, N0, N1, ... , Nm).
            denoiser_model (image_dimeras model instance): A pre-trained cnn denoiser model.
            data_format (string): One of ``channels_last``(default) or ``channles_first``. The ordering of the dimensions in the input image volume.
                ``channels_last`` corresponds to inputs with shape (batch_size, height, width, channels).
                ``channels_first`` corresponds to inputs with shape (batch_size, channels, height, width).
                
        Returns:
            ndarray: denoised image batch with shape (N_batch, N0, N1, ... , Nm).
    """
    valid_format = {"channels_last","channels_first"}
    assert data_format in valid_format, "Invalid data format."
    tensor_noisy = _to_tensor(img_noisy, data_format)
    tensor_denoised = denoiser_model.predict(tensor_noisy) # inference
    img_denoised = _from_tensor(tensor_denoised)
    return img_denoised

