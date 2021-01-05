
import numpy as np
from utils import *

x = read_recon3D('object.sino')
x = np.copy(np.swapaxes(x, 0, 2), order='C')
print(x.shape)
np.save('sino.npy', x)

x = read_recon3D('object.wght')
x = np.copy(np.swapaxes(x, 0, 2), order='C')
print(x.shape)
np.save('wght.npy', x)
