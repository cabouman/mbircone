
import numpy as np
import mbircone
import demo_utils

sino = np.load('sino.npy')
sino = np.copy(np.swapaxes(sino, 1, 2))
# Make shape : views x slices x channels
    
angles = np.linspace(0, 2*np.pi, 40, endpoint=False)


dist_source_detector = 619.7902
magnification = 8.651889263
delta_pixel_detector = 3.2
channel_offset = -0.9384
row_offset = 0.5654


x = mbircone.recon(sino, angles, dist_source_detector=dist_source_detector, magnification=magnification, 
	delta_pixel_detector=delta_pixel_detector,
	channel_offset=channel_offset, row_offset=row_offset, sharpness=1,
	max_iterations=10)



fname_ref = 'object.phantom.recon'
ref = demo_utils.read_ND(fname_ref, 3)
ref = np.swapaxes(ref, 0, 2)

rmse_val = np.sqrt(np.mean((x-ref)**2))
print("RMSE between reconstruction and reference: {}".format(rmse_val))

demo_utils.plot_image(x[60], title='recon', filename='output/recon_60.png', vmin=0, vmax=0.1)
demo_utils.plot_image(ref[60], title='ref', filename='output/ref_60.png', vmin=0, vmax=0.1)

demo_utils.plot_image(x[65], title='recon', filename='output/recon_65.png', vmin=0, vmax=0.1)
demo_utils.plot_image(ref[65], title='ref', filename='output/ref_65.png', vmin=0, vmax=0.1)
