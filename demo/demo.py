
import numpy as np
# from mbircone import AmatrixComputeToFile_cy
import mbircone


sino = np.load('sino.npy')
wght = np.load('wght.npy')
    
angles = np.linspace(0, 2*np.pi, 40, endpoint=False)

print(angles)


sinoparams = dict()
sinoparams['N_dv'] = 64
sinoparams['N_dw'] = 64
sinoparams['N_beta'] = 40
sinoparams['Delta_dv'] = 3.2
sinoparams['Delta_dw'] = 3.2
sinoparams['u_s'] = -71.6364
sinoparams['u_r'] = 0
sinoparams['v_r'] = 0
sinoparams['u_d0'] = 548.1538
sinoparams['v_d0'] = -101.4616
sinoparams['w_d0'] = -102.9654
sinoparams['weightScaler_value'] = -1

imgparams = dict()
imgparams['x_0'] = -11.9663
imgparams['y_0'] = -11.9663
imgparams['z_0'] = -14.7378
imgparams['N_x'] = 65
imgparams['N_y'] = 65
imgparams['N_z'] = 80
imgparams['Delta_xy'] = 0.36986
imgparams['Delta_z'] = 0.36986
imgparams['j_xstart_roi'] = 2
imgparams['j_ystart_roi'] = 2
imgparams['j_zstart_roi'] = 14
imgparams['j_xstop_roi'] = 62
imgparams['j_ystop_roi'] = 62
imgparams['j_zstop_roi'] = 66
imgparams['N_x_roi'] = 61
imgparams['N_y_roi'] = 61
imgparams['N_z_roi'] = 53

Amatrix_fname = 'test.sysmatrix'
mbircone.AmatrixComputeToFile_cy(angles, sinoparams, imgparams, Amatrix_fname, verbose=0)
