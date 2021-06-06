
import os
import numpy as np
# from mbircone import AmatrixComputeToFile_cy
import mbircone
import matplotlib.pyplot as plt

def read_ND(filePath, n_dim, dtype='float32', ntype='int32'):

    with open(filePath, 'rb') as fileID:

        sizesArray = np.fromfile( fileID, dtype=ntype, count=n_dim)
        numElements = np.prod(sizesArray)
        dataArray = np.fromfile(fileID, dtype=dtype, count=numElements).reshape(sizesArray)

    return dataArray

def plot_image(img, title=None, filename=None, vmin=None, vmax=None):
    """
    Function to display and save a 2D array as an image.

    Args:
        img: 2D numpy array to display
        title: Title of plot image
        filename: A path to save plot image
        vmin: Value mapped to black
        vmax: Value mapped to white
    """

    plt.ion()
    fig = plt.figure()
    imgplot = plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.title(label=title)
    imgplot.set_cmap('gray')
    plt.colorbar()

    if not os.path.exists(os.path.dirname(filename)):
        os.mkdir(os.path.dirname(filename))
    
    plt.savefig(filename)



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
reconparams['MaxIterations'] = 40
reconparams['relativeChangeMode'] = 'percentile'
reconparams['relativeChangeScaler'] = 0.1
reconparams['relativeChangePercentile'] = 99.9
reconparams['zipLineMode'] = 2
reconparams['N_G'] = 2
reconparams['numVoxelsPerZiplineMax'] = 200
reconparams['numVoxelsPerZipline'] = 200
reconparams['numZiplines'] = 4
reconparams['numThreads'] = 20
reconparams['weightScaler_domain'] = 'spatiallyInvariant'
reconparams['weightScaler_estimateMode'] = 'avgWghtRecon'
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
x_init = np.zeros((imgparams['N_x'],imgparams['N_y'],imgparams['N_z']))
proxmap_input = np.zeros((imgparams['N_x'],imgparams['N_y'],imgparams['N_z']))


print('Reconstructing ...')
x = mbircone.recon_cy(sino, wght, x_init, proxmap_input,
             sinoparams, imgparams, reconparams, Amatrix_fname)
print('Reconstructing done.')

Ax = mbircone.project_cy(x, sinoparams, imgparams, Amatrix_fname)

x = np.swapaxes(x, 0, 2)
Ax = np.swapaxes(Ax, 1, 2)
sino = np.swapaxes(sino, 1, 2)

fname_ref = 'object.phantom.recon'
ref = read_ND(fname_ref, 3)
ref = np.swapaxes(ref, 0, 2)

rmse_val = np.sqrt(np.mean((x-ref)**2))
print("RMSE between reconstruction and reference: {}".format(rmse_val))

plot_image(x[65], title='recon', filename='output/recon.png', vmin=0, vmax=0.1)
plot_image(ref[65], title='ref', filename='output/ref.png', vmin=0, vmax=0.1)

plot_image(sino[0]-Ax[0], title='errsino', filename='output/errsino.png')

