
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


x = mbircone.recon(sino, angles, 0, 0, weights=wght)


fname_ref = 'object.phantom.recon'
ref = demo_utils.read_ND(fname_ref, 3)
ref = np.swapaxes(ref, 0, 2)

rmse_val = np.sqrt(np.mean((x-ref)**2))
print("RMSE between reconstruction and reference: {}".format(rmse_val))
