import numpy as np
import os,sys
import time
import mbircone.cone3D as cone3D 

__lib_path = os.path.join(os.path.expanduser('~'), '.cache', 'mbircone')

def compute_inv_permute_vector(permute_vector):
    ''' Given a permutation vector, compute its inverse permutation vector s.t. an array will have the same shape after permutation and inverse permutation. 
    '''
    inv_permute_vector = []
    for i in range(len(permute_vector)):
        # print('i = {}'.format(i))
        position_of_i = permute_vector.index(i)
        # print('position_of_i = {}'.format(position_of_i))
        inv_permute_vector.append(position_of_i)
    return tuple(inv_permute_vector)


def normalize(img, image_range):
    """Normalizes ``img`` from specified image range to the range of (0,1).
    """
    #print('original image range:',image_range)
    img_normalized = (img-image_range[0])/(image_range[1]-image_range[0])
    #print('normalized image range:',np.percentile(img_normalized,10),np.percentile(img_normalized,90))
    return img_normalized


def denormalize(img_normalized, image_range):
    """Denormalizes ``img_normalized`` from (0,1) to desired image range.
    """
    img = img_normalized*(image_range[1]-image_range[0])+image_range[0] 
    return img


def denoiser_wrapper(image_noisy, denoiser, denoiser_args, image_range, permute_vector=(0,1,2), positivity=True):
    ''' This is a denoiser wrapper function. Given an image volume to be denoised, the wrapper function permutes and normalizes the image, passes it to a denoiser function, and permutes and denormalizes the denoised image back.
    Args:
        image_noisy (ndarray): image volume to be denoised
        denoiser (callable): The denoiser function to be used.
            denoiser(x, *denoiser_args) -> ndarray
            where ``x`` is an ndarray of the noisy image volume, and ``denoiser_args`` is a tuple of the fixed parameters needed to completely specify the denoiser function.
        denoiser_args (tuple): [Default=()] Extra arguments passed to the denoiser function.
        image_range (tuple): dynamic range of reconstruction image. 
        permute_vector (tuple): [Default=(0,1,2)] 
            It contains a permutation of [0,1,..,N-1] where N is the number of axes of image_noisy. The i’th axis of the permuted array will correspond to the axis numbered axes[i] of image_noisy. If not specified, defaults to (0,1,2), which effectively does no permutation.
        positivity: positivity constraint for denoiser output.
            If True, positivity will be enforced by clipping the denoiser output to be non-negative.
    Returns:
        ndarray: denoised image with same shape and dimensionality as input image ``image_noisy`` 
    '''
    # permute the 3D image s.t. the desired denoising dimensionality is moved to axis=0
    image_noisy = np.transpose(image_noisy, permute_vector)
    image_noisy_norm = normalize(image_noisy, image_range)
    # denoise!
    image_denoised_norm = denoiser(image_noisy_norm, denoiser_args) 
    # denormalize image from [0,1] to original dynamic range
    image_denoised = denormalize(image_denoised_norm, image_range)
    if positivity:
        image_denoised=np.clip(image_denoised, 0, None)
    # permute the denoised image back
    inv_permute_vector = compute_inv_permute_vector(permute_vector)
    image_denoised = np.transpose(image_denoised, inv_permute_vector)
    return image_denoised


def mace3D(sino, angles, dist_source_detector, magnification,
            denoiser, denoiser_args=(),
            max_admm_itr=10, rho=0.5, prior_weight=0.5,
            init_image=None, image_range=None,
            channel_offset=0.0, row_offset=0.0, rotation_offset=0.0,
            delta_pixel_detector=1.0, delta_pixel_image=None, ror_radius=None,
            sigma_y=None, snr_db=30.0, weights=None, weight_type='unweighted',
            positivity=True, p=1.2, q=2.0, T=1.0, num_neighbors=6,
            sharpness=0.0, sigma_x=None, sigma_p=None, max_iterations=3, stop_threshold=0.02,
            num_threads=None, NHICD=False, verbose=1, lib_path=__lib_path):
    """Computes 3D conebeam beam reconstruction with multi-slice MACE alogorithm by fusing forward model proximal map with 2D denoisers across xy, xz, and yz planes.
    Required Args:
        sino (ndarray): 3D sinogram array with shape (num_views, num_det_rows, num_det_channels)
        angles (ndarray): 1D view angles array in radians.
        dist_source_detector (float): Distance between the X-ray source and the detector in units of ALU
        magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance). 
    
    Args specific to MACE reconstruction algorithm:        
        denoiser (callable): The denoiser function used as the prior agent in MACE.
            denoiser(x, *denoiser_args) -> ndarray
            where ``x`` is an ndarray of the noisy image volume, and ``denoiser_args`` is a tuple of the fixed parameters needed to completely specify the denoiser function. 
        denoiser_args (tuple): [Default=()] Extra arguments passed to the denoiser function.
         
        max_admm_itr (int): [Default=10] Maximum number of MACE ADMM iterations.
        rho (float): [Default=0.5] step size of ADMM update in MACE, range (0,1).
            The value of rho mainly controls the convergence speed of MACE algorithm.
        prior_weight (ndarray): [Default=0.5] weights for prior agents, specified by either a scalar value or a 1D array.
            If a scalar is specified, then all three prior agents use the same weight of (prior_weight/3).
            If an array is provided, then the array should have three elements corresponding to the weight of denoisers in XY, YZ, and XZ planes respectively. 
            The weight for forward model proximal map agent will be calculated as 1-sum(prior_weight) so that the sum of all agent weights are equal to 1.
            Each entry of prior_weight should have value between 0 and 1. sum(prior_weight) needs to be no greater than 1.
        
        init_image: (ndarray, optional): [Default=None] Initial value of MACE reconstruction image, specified by either a scalar value or a 3D numpy array with shape (num_img_slices,num_img_rows,num_img_cols)
            If None, the inital value of MACE will be automatically determined by a qGGMRF reconstruction.
        image_range (tuple): [default: None] dynamic range of reconstruction image. 
            If None, the lower bound will be 0, and the upper bound will be determined by 95% pixel value of the qGGMRF reconstruction. 
            If an init_image is provided, then image_range must be also provided. 
        
    Optional Args inherited from ``mbircone.cone3D.recon`` (with changing default value of ``max_iterations``): 
        channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a row.
        row_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a column.
        rotation_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from source-detector line to axis of rotation in the object space.
            This is normally set to zero.

        delta_pixel_detector (float, optional): [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.
        delta_pixel_image (float, optional): [Default=None] Scalar value of image pixel spacing in :math:`ALU`.
            If None, automatically set to delta_pixel_detector/magnification
        ror_radius (float, optional): [Default=None] Scalar value of radius of reconstruction in :math:`ALU`.
            If None, automatically set with compute_img_params.
            Pixels outside the radius ror_radius in the :math:`(x,y)` plane are disregarded in the reconstruction.

        sigma_y (float, optional): [Default=None] Scalar value of noise standard deviation parameter.
            If None, automatically set with auto_sigma_y.
        snr_db (float, optional): [Default=30.0] Scalar value that controls assumed signal-to-noise ratio of the data in dB.
            Ignored if sigma_y is not None.
        weights (ndarray, optional): [Default=None] 3D weights array with same shape as sino.
        weight_type (string, optional): [Default='unweighted'] Type of noise model used for data.
            If the ``weights`` array is not supplied, then the function ``mbircone.calc_weights`` is used to set weights using specified ``weight_type`` parameter.
            Option "unweighted" corresponds to unweighted reconstruction;
            Option "transmission" is the correct weighting for transmission CT with constant dosage;
            Option "transmission_root" is commonly used with transmission CT data to improve image homogeneity;
            Option "emission" is appropriate for emission CT data.

        positivity (bool, optional): [Default=True] Boolean value that determines if positivity constraint is enforced.
            The positivity parameter defaults to True; however, it should be changed to False when used in applications that can generate negative image values.
        p (float, optional): [Default=1.2] Scalar value in range :math:`[1,2]` that specifies the qGGMRF shape parameter.
        q (float, optional): [Default=2.0] Scalar value in range :math:`[p,1]` that specifies the qGGMRF shape parameter.
        T (float, optional): [Default=1.0] Scalar value :math:`>0` that specifies the qGGMRF threshold parameter.
        num_neighbors (int, optional): [Default=6] Possible values are {26,18,6}.
            Number of neightbors in the qggmrf neighborhood. Higher number of neighbors result in a better regularization but a slower reconstruction.
        
        sharpness (float, optional): [Default=0.0]
            Scalar value that controls level of sharpness in the reconstruction
            ``sharpness=0.0`` is neutral; ``sharpness>0`` increases sharpness; ``sharpness<0`` reduces sharpness.
            Ignored in qGGMRF reconstruction if ``sigma_x`` is not None, or in proximal map estimation if ``sigma_p`` is not None.
        sigma_x (float, optional): [Default=None] Scalar value :math:`>0` that specifies the qGGMRF scale parameter.
            If None, automatically set with auto_sigma_x. Regularization should be controled with the ``sharpness`` parameter, but ``sigma_x`` can be set directly by expert users.
        sigma_p (float, optional): [Default=None] Scalar value :math:`>0` that specifies the proximal map parameter.
            If None, automatically set with auto_sigma_p. Regularization should be controled with the ``sharpness`` parameter, but ``sigma_p`` can be set directly by expert users.
        max_iterations (int, optional): [Default=3] Integer valued specifying the maximum number of iterations for proximal map estimation.
        stop_threshold (float, optional): [Default=0.0] [Default=0.02] Scalar valued stopping threshold in percent.
            If stop_threshold=0.0, then run max iterations.
        
        num_threads (int, optional): [Default=None] Number of compute threads requested when executed.
            If None, num_threads is set to the number of cores in the system
        NHICD (bool, optional): [Default=False] If true, uses Non-homogeneous ICD updates
        verbose (int, optional): [Default=1] Possible values are {0,1,2}, where 0 is quiet, 1 prints minimal reconstruction progress information, and 2 prints the full information.
        lib_path (str, optional): [Default=~/.cache/mbircone] Path to directory containing library of forward projection matrices.
    Returns:
        3D numpy array: 3D reconstruction with shape (num_img_slices, num_img_rows, num_img_cols) in units of :math:`ALU^{-1}`.        
    """
    
    print("initializing MACE...")
    if weights is None:
        weights = cone3D.calc_weights(sino,weight_type)
    if delta_pixel_image is None:
        delta_pixel_image = delta_pixel_detector/magnification
    # Set automatic value of sigma_y
    if sigma_y is None:
        sigma_y = cone3D.auto_sigma_y(sino, weights, snr_db=snr_db, delta_pixel_image=delta_pixel_image, delta_pixel_detector=delta_pixel_detector)
    # Set automatic value of sigma_p
    if sigma_p is None:
        sigma_p = cone3D.auto_sigma_p(sino, delta_pixel_detector=delta_pixel_detector, sharpness=sharpness)
    # Set automatic value of sigma_x
    if sigma_x is None:
        sigma_x = cone3D.auto_sigma_x(sino, delta_pixel_detector=delta_pixel_detector, sharpness=sharpness)
    if init_image is None:
        print('Computing qGGMRF recon. This will be use it as MACE initialization point.') 
        # variable initialization
        init_image = cone3D.recon(sino, angles, dist_source_detector, magnification,
              channel_offset=channel_offset, row_offset=row_offset, rotation_offset=rotation_offset,
              delta_pixel_detector=delta_pixel_detector, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius,
              sigma_y=sigma_y, weights=weights,
              positivity=positivity, p=p, q=q, T=T, num_neighbors=num_neighbors,
              sigma_x=sigma_x, stop_threshold=stop_threshold,
              num_threads=num_threads, NHICD=NHICD, verbose=verbose, lib_path=lib_path)
        if image_range is None:
            print("image dynamic range automatically determined by qGGMRF reconstruction.")
            image_range_upper = np.percentile(init_image, 95)
            image_range = [0, image_range_upper]
            print("image dynamic range = ",image_range)
    # Throw an exception if image_range is None and init_image is not None.
    assert not (image_range is None), \
        'Image_range needs to be provided if an init_image is given to MACE algorithm.'
        
    [Nz,Nx,Ny] = np.shape(init_image)
    image_dim = np.ndim(init_image)
    # number of agents = image dimensionality + 1.
    W = [np.copy(init_image) for _ in range(image_dim+1)]
    X = [np.copy(init_image) for _ in range(image_dim+1)]
    
    # agent weight
    if isinstance(prior_weight, (list, tuple, np.ndarray)):
        assert (len(prior_weight)==image_dim), 'Incorrect dimensionality of prior_weight array.'
        beta = [1-sum(prior_weight)]
        for w in prior_weight:
            beta.append(w)
    else:
        beta = [1-prior_weight,prior_weight/(image_dim),prior_weight/(image_dim),prior_weight/(image_dim)]
    assert(all(w>=0 for w in beta)), 'Incorrect value of prior_weight given. All elements in prior_weight should be non-negative, and sum should be no greater than 1.'   
    # begin ADMM iterations
    print("Begin MACE ADMM iterations")
    for itr in range(max_admm_itr):
        # forward model prox map agent
        start = time.time()
        X[0] = cone3D.recon(sino, angles, dist_source_detector, magnification,
          channel_offset=channel_offset, row_offset=row_offset, rotation_offset=rotation_offset,
          delta_pixel_detector=delta_pixel_detector, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius,
          init_image=X[0], prox_image=W[0],
          sigma_y=sigma_y, weights=weights,
          positivity=positivity,
          sigma_p=sigma_p, max_iterations=max_iterations, stop_threshold=stop_threshold,
          num_threads=num_threads, NHICD=NHICD, verbose=False, lib_path=lib_path)
        # prior model denoiser agents
        # denoising in XY plane (along Z-axis)
        X[1] = denoiser_wrapper(W[1], denoiser, denoiser_args, image_range, permute_vector=(0,1,2), positivity=positivity)
        # denoising in YZ plane (along X-axis)
        X[2] = denoiser_wrapper(W[2], denoiser, denoiser_args, image_range, permute_vector=(1,0,2), positivity=positivity)
        # denoising in XZ plane (along Y-axis)
        X[3] = denoiser_wrapper(W[3], denoiser, denoiser_args, image_range, permute_vector=(2,0,1), positivity=positivity) 
        Z = sum([beta[k]*(2*X[k]-W[k]) for k in range(image_dim+1)])
        for k in range(image_dim+1):
            W[k] += 2*rho*(Z-X[k])
        recon = sum([beta[k]*X[k] for k in range(image_dim+1)])
        end = time.time()
        elapsed_t = end-start
        print(f'MACE iteration {itr}, elapsed time: {elapsed_t:.2f} sec.')
    # end ADMM iterations
    return recon

