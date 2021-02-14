
import numpy as np
# from mbircone import AmatrixComputeToFile_cy
import mbircone


sino = np.load('sino.npy')
wght = np.load('wght.npy')
    
angles = np.linspace(0, 2*np.pi, 40, endpoint=False)

print(angles)

# mbircone.AmatrixComputeToFile_cy(angles)

