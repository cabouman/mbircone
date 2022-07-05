import math
from psutil import cpu_count
import shutil
import numpy as np
import os
import hashlib
import mbircone.interface_cy_c as ci
import random
import warnings

from mbircone.cone3D import hash_params, calc_weights, auto_sigma_y, auto_sigma_x, auto_sigma_p

"""
I have created a minimalist expansion upon cone3D without rewriting anything in preexisting code. The only changed *preexisting* module is __init__.py
All function signatures are in laminography coordinates.
Where cone3D methods lack functionality I wrote a 'laminography' method, which performs the role of that cone3D method instead of calling it. This resulted in redundant private functions as well as some spaghetti-fication.
The sinoparams dictionary is computed in an entirely different way than in cone3D. This is because the
Next step would be to consider refactoring, using this implementation as a rough prototype.
"""

__lib_path = os.path.join(os.path.expanduser('~'), '.cache', 'mbircone')
__namelen_sysmatrix = 20


def _gen_sysmatrix_fname(lib_path=__lib_path, sysmatrix_name='object'):
    os.makedirs(os.path.join(lib_path, 'sysmatrix'), exist_ok=True)

    sysmatrix_fname = os.path.join(lib_path, 'sysmatrix', sysmatrix_name + '.sysmatrix')

    return sysmatrix_fname


def _gen_sysmatrix_fname_tmp(lib_path=__lib_path, sysmatrix_name='object'):
    sysmatrix_fname_tmp = os.path.join(lib_path, 'sysmatrix',
                                       sysmatrix_name + '_pid' + str(os.getpid()) + '_rndnum' + str(
                                           random.randint(0, 1000)) + '.sysmatrix')

    return sysmatrix_fname_tmp


def compute_sino_params_lamino(num_views, num_det_rows, num_det_channels,
										theta,
										channel_offset=0.0,
										delta_pixel_detector=1.0):
	
	# This just needs to be large
	dist_factor = 500
	dist_source_detector = (dist_factor) * (delta_pixel_detector) * max(num_det_channels, num_det_rows)**2

	sinoparams = dict()
	sinoparams['N_dv'] = num_det_channels
	sinoparams['N_dw'] = num_det_rows
	sinoparams['N_beta'] = num_views
	sinoparams['Delta_dv'] = delta_pixel_detector
	sinoparams['Delta_dw'] = delta_pixel_detector / math.sin(theta)
	sinoparams['u_s'] = - dist_source_detector
	sinoparams['u_r'] = 0
	sinoparams['v_r'] = 0
	sinoparams['u_d0'] = 0

	dist_dv_to_detector_corner_from_detector_center = - sinoparams['N_dv'] * sinoparams['Delta_dv'] / 2
	dist_dw_to_detector_corner_from_detector_center = - sinoparams['N_dw'] * sinoparams['Delta_dw'] / 2

	dist_dv_to_detector_center_from_source_detector_line = - channel_offset
	dist_dw_to_detector_center_from_source_detector_line = - dist_source_detector / math.tan(theta)

	# corner of detector from source-detector-line
	sinoparams[
			'v_d0'] = dist_dv_to_detector_corner_from_detector_center + dist_dv_to_detector_center_from_source_detector_line
	sinoparams[
			'w_d0'] = dist_dw_to_detector_corner_from_detector_center + dist_dw_to_detector_center_from_source_detector_line

	sinoparams['weightScaler_value'] = -1

	return sinoparams

def compute_img_params_lamino(sinoparams, theta, delta_pixel_image, ror_radius):
	
	s = delta_pixel_image
	ell = delta_pixel_image
	
	imgparams = dict()
	imgparams['Delta_xy'] = s
	imgparams['Delta_z'] = ell
	
	N_C = sinoparams['N_dv']
	T_C = sinoparams['Delta_dv']
	N_R = sinoparams['N_dw']
	T_R = sinoparams['Delta_dw']
	
	R = (N_R * T_R) / (2 * np.cos(theta))
	H = (N_R * T_R) / (2 * np.sin(theta))
	
	y_0 = sinoparams['v_d0'] + (N_C * T_C / 2)
	r_cyl_1 = (N_C * T_C / 2) - np.abs(y_0)
	r_cyl_2 = (N_R * T_R / 2)
	r = min(r_cyl_1,r_cyl_2)
	
	h = H - (r / np.tan(theta))
	
	imgparams['x_0'] = - R - s / 2
	imgparams['y_0'] = imgparams['x_0']
	imgparams['z_0'] = - H + sinoparams['w_d0'] + (N_R * T_R / 2)
	
	imgparams['N_x'] = int(2 * np.ceil(R / s) + 1)
	imgparams['N_y'] = imgparams['N_x']
	imgparams['N_z'] = int(2 * np.ceil(H / ell) + 1)

	imgparams['j_xstart_roi'] = int(np.ceil(R/s) - np.floor(r/s))
	imgparams['j_ystart_roi'] = imgparams['j_xstart_roi']
	imgparams['j_zstart_roi'] = int(np.ceil(H/ell) - np.floor(h/ell))
	
	imgparams['j_xstop_roi'] = int(np.ceil(R/s) + np.floor(r/s))
	imgparams['j_ystop_roi'] = imgparams['j_xstop_roi']
	imgparams['j_zstop_roi'] = int(np.ceil(H/ell) + np.floor(h/ell))
	
	imgparams['N_x_roi'] = -1
	imgparams['N_y_roi'] = -1
	imgparams['N_z_roi'] = -1
	
	return imgparams


def compute_img_size_lamino(num_views, num_det_rows, num_det_channels,
								 theta,
								 channel_offset=0.0,
								 delta_pixel_detector=1.0, delta_pixel_image=None, ror_radius=None):


	# Automatically set delta_pixel_image.
	if delta_pixel_image is None:
			delta_pixel_image = delta_pixel_detector

	# Calculate parameter dictionary with given input.
	sinoparams = compute_sino_params_lamino(num_views=num_views, num_det_rows=num_det_rows, num_det_channels=num_det_channels,
																	 theta=theta,
																	 channel_offset=channel_offset,
																	 delta_pixel_detector=delta_pixel_detector)

	imgparams = compute_img_params_lamino(sinoparams, theta, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius)

	# Summarize Information about the image size.
	ROR = [imgparams['N_z'], imgparams['N_x'], imgparams['N_y']]
	boundary_size = [max(imgparams['j_zstart_roi'], imgparams['N_z']-1-imgparams['j_zstop_roi']), imgparams['j_xstart_roi'], imgparams['j_ystart_roi']]

	return ROR, boundary_size


def recon_lamino(sino, angles, theta,
			channel_offset=0.0,
			delta_pixel_detector=1.0, delta_pixel_image=None, ror_radius=None,
			init_image=0.0, prox_image=None,
			sigma_y=None, snr_db=40.0, weights=None, weight_type='unweighted',
			positivity=True, p=1.2, q=2.0, T=1.0, num_neighbors=6,
			sharpness=0.0, sigma_x=None, sigma_p=None, max_iterations=100, stop_threshold=0.02,
			num_threads=None, NHICD=False, verbose=1, lib_path=__lib_path):


	# Internally set
	# NHICD_ThresholdAllVoxels_ErrorPercent=80, NHICD_percentage=15, NHICD_random=20,
	# zipLineMode=2, N_G=2, numVoxelsPerZiplineMax=200

	if num_threads is None:
			num_threads = cpu_count(logical=False)

	os.environ['OMP_NUM_THREADS'] = str(num_threads)
	os.environ['OMP_DYNAMIC'] = 'true'

	if delta_pixel_image is None:
			delta_pixel_image = delta_pixel_detector

	(num_views, num_det_rows, num_det_channels) = sino.shape

	sinoparams = compute_sino_params_lamino(num_views=num_views, num_det_rows=num_det_rows, num_det_channels=num_det_channels,
																	 theta=theta,
																	 channel_offset=channel_offset,
																	 delta_pixel_detector=delta_pixel_detector)

	imgparams = compute_img_params_lamino(sinoparams, theta, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius)

	hash_val = hash_params(angles, sinoparams, imgparams)
	sysmatrix_fname = _gen_sysmatrix_fname(lib_path=lib_path, sysmatrix_name=hash_val[:__namelen_sysmatrix])

	if os.path.exists(sysmatrix_fname):
			os.utime(sysmatrix_fname)  # update file modified time
	else:
			sysmatrix_fname_tmp = _gen_sysmatrix_fname_tmp(lib_path=lib_path, sysmatrix_name=hash_val[:__namelen_sysmatrix])
			ci.AmatrixComputeToFile_cy(angles, sinoparams, imgparams, sysmatrix_fname_tmp, verbose=verbose)
			os.rename(sysmatrix_fname_tmp, sysmatrix_fname)
	# make sure that weights do not contain negative entries
	# if weights is provided, and negative entry exists, then do not use the provided weights
	if not ((weights is None) or (np.amin(weights) >= 0.0)):
			warnings.warn("Parameter weights contains negative values; Setting weights = None.")
			weights = None
	# Set automatic values for weights
	if weights is None:
			weights = calc_weights(sino, weight_type)

	# Set automatic value of sigma_y
	if sigma_y is None:
			sigma_y = auto_sigma_y(sino, 1, weights, snr_db,
														 delta_pixel_image=delta_pixel_image,
														 delta_pixel_detector=delta_pixel_detector)

	# Set automatic value of sigma_x
	if sigma_x is None:
			sigma_x = auto_sigma_x(sino, 1, delta_pixel_detector=delta_pixel_detector, sharpness=sharpness)


	reconparams = dict()
	reconparams['is_positivity_constraint'] = bool(positivity)
	reconparams['q'] = q
	reconparams['p'] = p
	reconparams['T'] = T
	reconparams['sigmaX'] = sigma_x

	if num_neighbors not in [6, 18, 26]:
			num_neighbors = 6

	if num_neighbors == 6:
			reconparams['bFace'] = 1.0
			reconparams['bEdge'] = -1
			reconparams['bVertex'] = -1

	if num_neighbors == 18:
			reconparams['bFace'] = 1.0
			reconparams['bEdge'] = 0.70710678118
			reconparams['bVertex'] = -1

	if num_neighbors == 26:
			reconparams['bFace'] = 1.0
			reconparams['bEdge'] = 0.70710678118
			reconparams['bVertex'] = 0.57735026919

	reconparams['stopThresholdChange_pct'] = stop_threshold
	reconparams['MaxIterations'] = max_iterations

	reconparams['weightScaler_value'] = sigma_y ** 2

	reconparams['verbosity'] = verbose

	################ Internally set

	# Weight scalar
	reconparams['weightScaler_domain'] = 'spatiallyInvariant'
	reconparams['weightScaler_estimateMode'] = 'None'

	# Stopping
	reconparams['stopThesholdRWFE_pct'] = 0
	reconparams['stopThesholdRUFE_pct'] = 0
	reconparams['relativeChangeMode'] = 'meanImage'
	reconparams['relativeChangeScaler'] = 0.1
	reconparams['relativeChangePercentile'] = 99.9

	# Zipline
	reconparams['zipLineMode'] = 2
	reconparams['N_G'] = 2
	reconparams['numVoxelsPerZiplineMax'] = 200
	reconparams['numVoxelsPerZipline'] = 200
	reconparams['numZiplines'] = 4

	# NHICD
	reconparams['NHICD_ThresholdAllVoxels_ErrorPercent'] = 80
	reconparams['NHICD_percentage'] = 15
	reconparams['NHICD_random'] = 20
	reconparams['isComputeCost'] = 1
	if NHICD:
			reconparams['NHICD_Mode'] = 'percentile+random'
	else:
			reconparams['NHICD_Mode'] = 'off'

	if prox_image is None:
			reconparams['prox_mode'] = False
			reconparams['sigma_lambda'] = 1
	else:
			reconparams['prox_mode'] = True
			if sigma_p is None:
					sigma_p = auto_sigma_p(sino, magnification, delta_pixel_detector, sharpness)
			reconparams['sigma_lambda'] = sigma_p

	x = ci.recon_cy(sino, weights, init_image, prox_image,
									sinoparams, imgparams, reconparams, sysmatrix_fname, num_threads)
	return x


def project_lamino(image, angles, theta,
				num_det_rows, num_det_channels,
				channel_offset=0.0,
				delta_pixel_detector=1.0, delta_pixel_image=None, ror_radius=None,
				num_threads=None, verbose=1, lib_path=__lib_path):
	

	if num_threads is None:
			num_threads = cpu_count(logical=False)

	os.environ['OMP_NUM_THREADS'] = str(num_threads)
	os.environ['OMP_DYNAMIC'] = 'true'

	if delta_pixel_image is None:
			delta_pixel_image = delta_pixel_detector

	num_views = len(angles)

	sinoparams = compute_sino_params_lamino(num_views=num_views, num_det_rows=num_det_rows, num_det_channels=num_det_channels,
																	 theta=theta,
																	 channel_offset=channel_offset,
																	 delta_pixel_detector=delta_pixel_detector)

	imgparams = compute_img_params_lamino(sinoparams, theta, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius)

	(num_img_slices, num_img_rows, num_img_cols) = image.shape

	assert (num_img_slices, num_img_rows, num_img_cols) == (imgparams['N_z'], imgparams['N_x'], imgparams['N_y']), \
			'Image size of %s is incorrect! With the specified geometric parameters, expected image should have shape %s, use function `cone3D.compute_img_size` to compute the correct image size.' \
			%  ((num_img_slices, num_img_rows, num_img_cols), (imgparams['N_z'], imgparams['N_x'], imgparams['N_y']))

	hash_val = hash_params(angles, sinoparams, imgparams)
	sysmatrix_fname = _gen_sysmatrix_fname(lib_path=lib_path, sysmatrix_name=hash_val[:__namelen_sysmatrix])

	if os.path.exists(sysmatrix_fname):
			os.utime(sysmatrix_fname)  # update file modified time
	else:
			sysmatrix_fname_tmp = _gen_sysmatrix_fname_tmp(lib_path=lib_path, sysmatrix_name=hash_val[:__namelen_sysmatrix])
			ci.AmatrixComputeToFile_cy(angles, sinoparams, imgparams, sysmatrix_fname_tmp, verbose=verbose)
			os.rename(sysmatrix_fname_tmp, sysmatrix_fname)

	# Collect settings to pass to C
	settings = dict()
	settings['imgparams'] = imgparams
	settings['sinoparams'] = sinoparams
	settings['sysmatrix_fname'] = sysmatrix_fname
	settings['num_threads'] = num_threads

	proj = ci.project(image, settings)
	return proj
