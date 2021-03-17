
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

reconparams = dict()
reconparams['InitVal_recon'] = 0
reconparams['initReconMode'] = 'constant'
reconparams['priorWeight_QGGMRF'] = 1
reconparams['priorWeight_proxMap'] = -1
reconparams['is_positivity_constraint'] = 1
reconparams['q'] = 2
reconparams['p'] = 1
reconparams['T'] = 0.02
reconparams['sigmaX'] = 5
reconparams['bFace'] = 1.0
reconparams['bEdge'] = 0.70710678118
reconparams['bVertex'] = 0.57735026919
reconparams['sigma_lambda'] = 1
reconparams['stopThresholdChange_pct'] = 0.00
reconparams['stopThesholdRWFE_pct'] = 0
reconparams['stopThesholdRUFE_pct'] = 0
reconparams['MaxIterations'] = 10
reconparams['relativeChangeMode'] = 'percentile'
reconparams['relativeChangeScaler'] = 0.1
reconparams['relativeChangePercentile'] = 99.9
reconparams['zipLineMode'] = 2
reconparams['N_G'] = 2
reconparams['numVoxelsPerZiplineMax'] = 200
reconparams['numVoxelsPerZipline'] = 200
reconparams['numZiplines'] = 4
reconparams['numThreads'] = 20
reconparams['weightScaler_domain'] = 'spatiallyVariant'
reconparams['weightScaler_estimateMode'] = 'None'
reconparams['weightScaler_value'] = 1
reconparams['NHICD_Mode'] = 'off'
reconparams['NHICD_ThresholdAllVoxels_ErrorPercent'] = 80
reconparams['NHICD_percentage'] = 15
reconparams['NHICD_random'] = 20
reconparams['verbosity'] = 0
reconparams['isComputeCost'] = 0
reconparams['backprojlike_type'] = 'proj'



Amatrix_fname = 'test.sysmatrix'
mbircone.AmatrixComputeToFile_cy(angles, sinoparams, imgparams, Amatrix_fname, verbose=1)
x = np.zeros((imgparams['N_x'],imgparams['N_y'],imgparams['N_z']))
x_init = np.zeros((imgparams['N_x'],imgparams['N_y'],imgparams['N_z']))
proxmap_input = np.zeros((imgparams['N_x'],imgparams['N_y'],imgparams['N_z']))

x_init[0,0,0] = 100
print(x_init[0,0,0])

mbircone.recon_cy(x, sino, wght, x_init, proxmap_input,
             sinoparams, imgparams, reconparams, Amatrix_fname)

print(x_init[0,0,0])
print(x[0,0,0])
