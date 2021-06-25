
import os
import numpy as np
import mbircone
import demo_utils

sino = np.load('sino.npy')
sino = np.copy(np.swapaxes(sino, 1, 2))
# Make shape : views x slices x channels
    
angles = np.linspace(0, 2*np.pi, 40, endpoint=False)


# magnification is unitless.
magnification = 8.651889263

# All distances are in unit of 1 ALU = 1 mm.
dist_source_detector = 619.7902
delta_pixel_detector = 3.2
channel_offset = -0.9384
row_offset = 0.5654


x = mbircone.recon(sino, angles, dist_source_detector=dist_source_detector, magnification=magnification, 
	delta_pixel_detector=delta_pixel_detector,
	channel_offset=channel_offset, row_offset=row_offset, sharpness=1,
	max_iterations=10)



p = mbircone.project(angles, x,
	num_slices=sino.shape[1], num_channels=sino.shape[2],
	dist_source_detector=dist_source_detector, magnification=magnification, 
	delta_pixel_detector=delta_pixel_detector,
	channel_offset=channel_offset, row_offset=row_offset)


# create output folder
os.makedirs('output', exist_ok=True)

demo_utils.plot_image(x[60], title='recon', filename='output/recon_60.png', vmin=0, vmax=0.1)
demo_utils.plot_image(x[65], title='recon', filename='output/recon_65.png', vmin=0, vmax=0.1)


demo_utils.plot_image(p[0], title='proj', filename='output/projection.png')
demo_utils.plot_image(sino[0], title='sino', filename='output/sino.png')
