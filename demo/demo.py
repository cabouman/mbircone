
import numpy as np
import mbircone
import demo_utils

sino = np.load('sino.npy')
wght = np.load('wght.npy')

sino = np.copy(np.swapaxes(sino, 1, 2))
wght = np.copy(np.swapaxes(wght, 1, 2))

# New shape : views x slices x channels
print(sino.shape)
    
angles = np.linspace(0, 2*np.pi, 40, endpoint=False)
print(angles)


dist_source_detector = 619.7902
magnification = 8.651889263
delta_pixel_detector = 3.2
delta_pixel_image = 0.36986
channel_offset = -0.9384
row_offset = 0.5654


x = mbircone.recon(sino, angles, dist_source_detector=dist_source_detector, magnification=magnification, 
	delta_pixel_detector=delta_pixel_detector, delta_pixel_image=delta_pixel_image,
	channel_offset=channel_offset, row_offset=row_offset,
	weights=wght, verbose=0)


fname_ref = 'object.phantom.recon'
ref = demo_utils.read_ND(fname_ref, 3)
ref = np.swapaxes(ref, 0, 2)

rmse_val = np.sqrt(np.mean((x-ref)**2))
print("RMSE between reconstruction and reference: {}".format(rmse_val))
