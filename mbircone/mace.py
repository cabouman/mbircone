import numpy as np
import os
import time
import mbircone.cone3D as cone3D
import mbircone.multinode as multinode

__lib_path = os.path.join(os.path.expanduser('~'), '.cache', 'mbircone')


def compute_inv_permute_vector(permute_vector):
    """ Given a permutation vector, compute its inverse permutation vector s.t. an array will have the same shape after permutation and inverse permutation. 
    """
     
    inv_permute_vector = []
    for i in range(len(permute_vector)):
        # print('i = {}'.format(i))
        position_of_i = permute_vector.index(i)
        # print('position_of_i = {}'.format(position_of_i))
        inv_permute_vector.append(position_of_i)
    return tuple(inv_permute_vector)


def denoiser_wrapper(image_noisy, denoiser, denoiser_args, permute_vector, positivity=True):
    """ This is a denoiser wrapper function. Given an image volume to be denoised, the wrapper function permutes and normalizes the image, passes it to a denoiser function, and permutes and denormalizes the denoised image back.

    Args:
        image_noisy (ndarray): image volume to be denoised
        denoiser (callable): The denoiser function to be used.

            ``denoiser(x, *denoiser_args) -> ndarray``

            where ``x`` is an ndarray of the noisy image volume, and ``denoiser_args`` is a tuple of the fixed parameters needed to completely specify the denoiser function.
        denoiser_args (tuple): [Default=()] Extra arguments passed to the denoiser function.
        permute_vector (tuple): permutation on the noisy image before passing to the denoiser function.
            It contains a permutation of [0,1,..,N-1] where N is the number of axes of image_noisy. The i’th axis of the permuted array will correspond to the axis numbered axes[i] of image_noisy. If not specified, defaults to (0,1,2), which effectively does no permutation.
            An inverse permutation is performed on the denoised image to make sure that the returned image has the same shape as the input noisy image.
        positivity: positivity constraint for denoiser output.
            If True, positivity will be enforced by clipping the denoiser output to be non-negative.

    Returns:
        ndarray: denoised image with same shape and dimensionality as input image ``image_noisy``
    """
    # permute the 3D image s.t. the desired denoising dimensionality is moved to axis=0
    image_noisy = np.transpose(image_noisy, permute_vector)
    # denoise
    image_denoised = denoiser(image_noisy, *denoiser_args)
    if positivity:
        image_denoised=np.clip(image_denoised, 0, None)
    # permute the denoised image back
    inv_permute_vector = compute_inv_permute_vector(permute_vector)
    image_denoised = np.transpose(image_denoised, inv_permute_vector)
    return image_denoised


def mace3D(sino, angles, dist_source_detector, magnification,
            denoiser, denoiser_args=(),
            max_admm_itr=10, rho=0.5, prior_weight=0.5,
            init_image=None, save_path=None,
            channel_offset=0.0, row_offset=0.0, rotation_offset=0.0,
            delta_pixel_detector=1.0, delta_pixel_image=None, ror_radius=None,
            sigma_y=None, snr_db=30.0, weights=None, weight_type='unweighted',
            positivity=True, p=1.2, q=2.0, T=1.0, num_neighbors=6,
            sharpness=0.0, sigma_x=None, sigma_p=None, max_iterations=3, stop_threshold=0.02,
            num_threads=None, NHICD=False, verbose=1, 
            lib_path=__lib_path):
    """Computes 3-D conebeam beam reconstruction with multi-slice MACE alogorithm by fusing forward model proximal map with 2D denoisers across xy, xz, and yz planes.
    
    Required arguments: 
        - **sino** (*ndarray*): 3-D sinogram array with shape (num_views, num_det_rows, num_det_channels)
        - **angles** (*ndarray*): 1D view angles array in radians.
        - **dist_source_detector** (*float*): Distance between the X-ray source and the detector in units of ALU
        - **magnification** (*float*): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance). 
    Arguments specific to MACE reconstruction algorithm:  
        - **denoiser** (*callable*): The denoiser function used as the prior agent in MACE.
            
                ``denoiser(x, *denoiser_args) -> ndarray``
            
            where ``x`` is an ndarray of the noisy image volume, and ``denoiser_args`` is a tuple of the fixed parameters needed to completely specify the denoiser function. 
        - **denoiser_args** (*tuple*): [Default=()] Extra arguments passed to the denoiser function.
        - **max_admm_itr** (*int*): [Default=10] Maximum number of MACE ADMM iterations.
        - **rho** (*float*): [Default=0.5] step size of ADMM update in MACE, range (0,1). The value of ``rho`` mainly controls the convergence speed of MACE algorithm.
        - **prior_weight** (*ndarray*): [Default=0.5] weights for prior agents, specified by either a scalar value or a 1D array. If a scalar is specified, then all three prior agents use the same weight of (prior_weight/3). If an array is provided, then the array should have three elements corresponding to the weight of denoisers in XY, YZ, and XZ planes respectively. The weight for forward model proximal map agent will be calculated as 1-sum(prior_weight) so that the sum of all agent weights are equal to 1. Each entry of prior_weight should have value between 0 and 1. sum(prior_weight) needs to be no greater than 1.
        - **init_image** (*ndarray, optional*): [Default=None] Initial value of MACE reconstruction image, specified by either a scalar value or a 3-D numpy array with shape (num_img_slices,num_img_rows,num_img_cols). If None, the inital value of MACE will be automatically determined by a qGGMRF reconstruction.
        - **save_path** (*str, optional*): [Default=None] Path to directory that saves the intermediate results of MACE.
    Optional arguments inherited from ``cone3D.recon`` (with changing default value of ``max_iterations``): 
        - **channel_offset** (*float, optional*): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a row.
        - **row_offset** (*float, optional*): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a column.
        - **rotation_offset** (*float, optional*): [Default=0.0] Distance in :math:`ALU` from source-detector line to axis of rotation in the object space. This is normally set to zero.
        - **delta_pixel_detector** (*float, optional*): [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.
        - **delta_pixel_image** (*float, optional*): [Default=None] Scalar value of image pixel spacing in :math:`ALU`. If None, automatically set to delta_pixel_detector/magnification
        - **ror_radius** (*float, optional*): [Default=None] Scalar value of radius of reconstruction in :math:`ALU`. If None, automatically set with compute_img_params. Pixels outside the radius ror_radius in the :math:`(x,y)` plane are disregarded in the reconstruction.
        - **sigma_y** (*float, optional*): [Default=None] Scalar value of noise standard deviation parameter. If None, automatically set with auto_sigma_y.
        - **snr_db** (*float, optional*): [Default=30.0] Scalar value that controls assumed signal-to-noise ratio of the data in dB. Ignored if sigma_y is not None.
        - **weights** (*ndarray, optional*): [Default=None] 3-D weights array with same shape as sino.
        - **weight_type** (*string, optional*): [Default='unweighted'] Type of noise model used for data. If the ``weights`` array is not supplied, then the function ``cone3D.calc_weights`` is used to set weights using specified ``weight_type`` parameter.
                
                - Option "unweighted" corresponds to unweighted reconstruction;
                - Option "transmission" is the correct weighting for transmission CT with constant dosage;
                - Option "transmission_root" is commonly used with transmission CT data to improve image homogeneity;
                - Option "emission" is appropriate for emission CT data.
        - **positivity** (*bool, optional*): [Default=True] Boolean value that determines if positivity constraint is enforced. The positivity parameter defaults to True; however, it should be changed to False when used in applications that can generate negative image values.
        - **p** (*float, optional*): [Default=1.2] Scalar value in range :math:`[1,2]` that specifies the qGGMRF shape parameter.
        - **q** (*float, optional*): [Default=2.0] Scalar value in range :math:`[p,1]` that specifies the qGGMRF shape parameter.
        - **T** (*float, optional*): [Default=1.0] Scalar value :math:`>0` that specifies the qGGMRF threshold parameter.
        - **num_neighbors** (*int, optional*): [Default=6] Possible values are {26,18,6}. Number of neightbors in the qggmrf neighborhood. Higher number of neighbors result in a better regularization but a slower reconstruction.
        - **sharpness** (*float, optional*): [Default=0.0] Scalar value that controls level of sharpness in the reconstruction. ``sharpness=0.0`` is neutral; ``sharpness>0`` increases sharpness; ``sharpness<0`` reduces sharpness. Ignored in qGGMRF reconstruction if ``sigma_x`` is not None, or in proximal map estimation if ``sigma_p`` is not None.
        - **sigma_x** (*float, optional*): [Default=None] Scalar value :math:`>0` that specifies the qGGMRF scale parameter. If None, automatically set with auto_sigma_x. Regularization should be controled with the ``sharpness`` parameter, but ``sigma_x`` can be set directly by expert users.
        - **sigma_p** (*float, optional*): [Default=None] Scalar value :math:`>0` that specifies the proximal map parameter. If None, automatically set with auto_sigma_p. Regularization should be controled with the ``sharpness`` parameter, but ``sigma_p`` can be set directly by expert users.
        - **max_iterations** (*int, optional*): [Default=3] Integer valued specifying the maximum number of iterations for proximal map estimation.
        - **stop_threshold** (*float, optional*): [Default=0.02] Scalar valued stopping threshold in percent. If stop_threshold=0.0, then run max iterations.
        - **num_threads** (*int, optional*): [Default=None] Number of compute threads requested when executed. If None, num_threads is set to the number of cores in the system
        - **NHICD** (*bool, optional*): [Default=False] If true, uses Non-homogeneous ICD updates
        - **verbose** (*int, optional*): [Default=1] Possible values are {0,1,2}, where 0 is quiet, 1 prints MACE reconstruction progress information, and 2 prints the MACE reconstruction as well as qGGMRF/proximal-map reconstruction progress information.
        - **lib_path** (*str, optional*): [Default=~/.cache/mbircone] Path to directory containing library of forward projection matrices.
    
    Returns:
        3-D numpy array: 3-D reconstruction with shape (num_img_slices, num_img_rows, num_img_cols) in units of :math:`ALU^{-1}`.        
    """
    
    if verbose: 
        print("initializing MACE...")
    # verbosity level for qGGMRF recon
    qGGMRF_verbose = max(0,verbose-1)     
    # Calculate automatic value of sinogram weights
    if weights is None:
        weights = cone3D.calc_weights(sino,weight_type)
    # Calculate automatic value of delta_pixel_image
    if delta_pixel_image is None:
        delta_pixel_image = delta_pixel_detector/magnification
    # Calculate automatic value of sigma_y
    if sigma_y is None:
        sigma_y = cone3D.auto_sigma_y(sino, weights, snr_db=snr_db, delta_pixel_image=delta_pixel_image, delta_pixel_detector=delta_pixel_detector)
    # Calculate automatic value of sigma_p
    if sigma_p is None:
        sigma_p = cone3D.auto_sigma_p(sino, delta_pixel_detector=delta_pixel_detector, sharpness=sharpness)
    if init_image is None:
        if verbose:
            start = time.time()
            print("Computing qGGMRF reconstruction. This will be used as MACE initialization point.") 
        init_image = cone3D.recon(sino, angles, dist_source_detector, magnification,
                                  channel_offset=channel_offset, row_offset=row_offset, rotation_offset=rotation_offset,
                                  delta_pixel_detector=delta_pixel_detector, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius,
                                  weights=weights, sigma_y=sigma_y, sigma_x=sigma_x,
                                  positivity=positivity, p=p, q=q, T=T, num_neighbors=num_neighbors,
                                  stop_threshold=stop_threshold,
                                  num_threads=num_threads, NHICD=NHICD, verbose=qGGMRF_verbose, lib_path=lib_path)
        if verbose:
            end = time.time()
            elapsed_t = end-start
            print(f"Done computing qGGMRF reconstruction. Elapsed time: {elapsed_t:.2f} sec.")
            if not (save_path is None): 
                print("Save qGGMRF reconstruction to disk.")
                np.save(os.path.join(save_path, 'recon_qGGMRF.npy'), init_image) 
       
    if np.isscalar(init_image):
        (num_views, num_det_rows, num_det_channels) = np.shape(sino)
        [Nz,Nx,Ny], _ = cone3D.compute_img_size(num_views, num_det_rows, num_det_channels, 
                                                    dist_source_detector, magnification, 
                                                    channel_offset=channel_offset, row_offset=row_offset, rotation_offset=rotation_offset, 
                                                    delta_pixel_detector=delta_pixel_detector, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius)
        init_image = np.zeros((Nz, Nx, Ny)) + init_image 
    else:
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
    # check that agent weights are all non-negative and sum up to 1.
    assert(all(w>=0 for w in beta) and (sum(beta)-1.)<1e-5), 'Incorrect value of prior_weight given. All elements in prior_weight should be non-negative, and sum should be no greater than 1.'
    # make denoiser_args an instance if necessary
    if not isinstance(denoiser_args, tuple):
        denoiser_args = (denoiser_args,) 
    
    ######################## begin ADMM iterations ########################
    if verbose:
        print("Begin MACE ADMM iterations:")
    for itr in range(max_admm_itr):
        if verbose:
            print(f"Begin MACE iteration {itr}/{max_admm_itr}:")
            itr_start = time.time()
        # forward model prox map agent
        X[0] = cone3D.recon(sino, angles, dist_source_detector, magnification,
                            channel_offset=channel_offset, row_offset=row_offset, rotation_offset=rotation_offset,
                            delta_pixel_detector=delta_pixel_detector, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius,
                            init_image=X[0], prox_image=W[0],
                            sigma_y=sigma_y, weights=weights,
                            positivity=positivity,
                            sigma_p=sigma_p, max_iterations=max_iterations, stop_threshold=stop_threshold,
                            num_threads=num_threads, NHICD=NHICD, verbose=qGGMRF_verbose, lib_path=lib_path)
        if verbose:
            print("Done forward model proximal map estimation.")
        # prior model denoiser agents
        denoise_start = time.time()
        # denoising in XY plane (along Z-axis)
        X[1] = denoiser_wrapper(W[1], denoiser, denoiser_args, permute_vector=(0,1,2), positivity=positivity)
        # denoising in YZ plane (along X-axis)
        X[2] = denoiser_wrapper(W[2], denoiser, denoiser_args, permute_vector=(1,0,2), positivity=positivity)
        # denoising in XZ plane (along Y-axis)
        X[3] = denoiser_wrapper(W[3], denoiser, denoiser_args, permute_vector=(2,0,1), positivity=positivity) 
        denoise_end = time.time()
        if verbose:
            denoise_elapsed = denoise_end - denoise_start
            print(f"Done denoising in all hyper-planes, elapsed time {denoise_elapsed:.2f} sec")
        if not (save_path is None):
            for i in range(4):
                np.save(os.path.join(save_path, f'X{i}_itr{itr}.npy'), X[i])
                np.save(os.path.join(save_path, f'W{i}_itr{itr}.npy'), W[i])

        Z = sum([beta[k]*(2*X[k]-W[k]) for k in range(image_dim+1)])
        for k in range(image_dim+1):
            W[k] += 2*rho*(Z-X[k])
        recon = sum([beta[k]*X[k] for k in range(image_dim+1)])
        if verbose:
            itr_end = time.time()
            itr_elapsed = itr_end-itr_start
            print(f"Done MACE iteration. Elapsed time: {itr_elapsed:.2f} sec.")
    ######################## end ADMM iterations ########################
    print("Done MACE reconstruction.")
    return recon


def mace4D(sino, angles, dist_source_detector, magnification,
           denoiser, denoiser_args=(),
           max_admm_itr=10, rho=0.5, prior_weight=0.5,
           init_image=None, save_path=None, 
           cluster_ticket=None,
           channel_offset=0.0, row_offset=0.0, rotation_offset=0.0,
           delta_pixel_detector=1.0, delta_pixel_image=None, ror_radius=None,
           sigma_y=None, snr_db=30.0, weights=None, weight_type='unweighted',
           positivity=True, p=1.2, q=2.0, T=1.0, num_neighbors=6,
           sharpness=0.0, sigma_x=None, sigma_p=None, max_iterations=3, stop_threshold=0.02,
           num_threads=None, NHICD=False, verbose=1, 
           lib_path=__lib_path):
    """Computes 4-D conebeam beam reconstruction with multi-slice MACE alogorithm by fusing forward model proximal map with 2.5-D denoisers across XY-t, XZ-t, and YZ-t hyperplanes.

    Required arguments:
        - **sino** (*list[ndarray]*): list of 3-D sinogram array. The length of sino is equal to num_time_points, where sino[t] is a 3-D array with shape (num_views, num_det_rows, num_det_channels), specifying sinogram of time point t.
        - **angles** (*list[list]*): List of view angles in radians. The length of angles is equal to num_time_points, where angles[t] is a 1D array specifying view angles of time point t. 
        - **dist_source_detector** (*float*): Distance between the X-ray source and the detector in units of ALU
        - **magnification** (*float*): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
    Arguments specific to MACE reconstruction algorithm:
        - **denoiser** (*callable*): The denoiser function used as the prior agent in MACE. 

                ``denoiser(x, *denoiser_args) -> ndarray``

            where ``x`` is a 4-D array of the noisy image volume with shape :math:`(N_{batch}, N_t, N_1, N_2)`, where the 2.5-D denoising hyper-plane is defined by :math:`(N_t, N_1, N_2)`. 

            ``denoiser_args`` is a tuple of the fixed parameters needed to completely specify the denoiser function. 
            
            The denoiser function should return a 4-D array of the denoised image, where the shape of the denoised image volume is the same as shape of the noisy image volume ``x``. 

            The same ``denoiser`` function is used to by all three denoising agents corresponding to XY-t, YZ-t and XZ-t hyperplanes. For each of the three denoising agents in MACE4D, the input noisy image volume will be permuted before fed into ``denoiser``, s.t. after permutation, :math:`(N_t, N_1, N_2)` corresponds to the denoising hyper-plane of the agent. 

        - **denoiser_args** (*tuple*): [Default=()] Extra arguments passed to the denoiser function.
        - **max_admm_itr** (*int*): [Default=10] Maximum number of MACE ADMM iterations.
        - **rho** (*float*): [Default=0.5] step size of ADMM update in MACE, range (0,1). The value of ``rho`` mainly controls the convergence speed of MACE algorithm.
        - **prior_weight** (*ndarray*): [Default=0.5] weights for prior agents, specified by either a scalar value or a 1D array. If a scalar is specified, then all three prior agents use the same weight of (prior_weight/3). If an array is provided, then the array should have three elements corresponding to the weight of denoisers in XY-t, YZ-t, and XZ-t hyperplanes respectively. The weight for forward model proximal map agent will be calculated as 1-sum(prior_weight) so that the sum of all agent weights are equal to 1. Each entry of prior_weight should have value between 0 and 1. sum(prior_weight) needs to be no greater than 1.
        - **init_image** (*ndarray, optional*): [Default=None] Initial value of MACE reconstruction image, specified by either a scalar value, a 3-D numpy array with shape (num_img_slices,num_img_rows,num_img_cols), or a 4-D numpy array with shape (num_time_points, num_img_slices,num_img_rows,num_img_cols). If None, the inital value of MACE will be automatically determined by a stack of 3-D qGGMRF reconstructions at different time points.
        - **save_path** (*str, optional*): [Default=None] Path to directory that saves the intermediate results of MACE. If not None, the inital qGGMRF reconstruction and input/output images of all agents from each MACE iteration will be saved to ``save_path``. 
    Arguments specific to multi-node computation:
        - **cluster_ticket** (*Object*): [Default=None] A ticket used to access a specific cluster, that can be obtained from ``multinode.get_cluster_ticket``. If cluster_ticket is None, the process will run in serial. See `dask_jobqueue <https://jobqueue.dask.org/en/latest/api.html>`_ for more information.
    Optional arguments inherited from ``cone3D.recon`` (with changing default value of ``max_iterations``):
        - **channel_offset** (*float, optional*): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a row.
        - **row_offset** (*float, optional*): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a column.
        - **rotation_offset** (*float, optional*): [Default=0.0] Distance in :math:`ALU` from source-detector line to axis of rotation in the object space. This is normally set to zero.
        - **delta_pixel_detector** (*float, optional*): [Default=1.0] Scalar value of detector pixel spacing in :math:`ALU`.
        - **delta_pixel_image** (*float, optional*): [Default=None] Scalar value of image pixel spacing in :math:`ALU`. If None, automatically set to delta_pixel_detector/magnification
        - **ror_radius** (*float, optional*): [Default=None] Scalar value of radius of reconstruction in :math:`ALU`. If None, automatically set with compute_img_params. Pixels outside the radius ror_radius in the :math:`(x,y)` plane are disregarded in the reconstruction.
        - **sigma_y** (*float, optional*): [Default=None] Scalar value of noise standard deviation parameter. If None, automatically set with auto_sigma_y.
        - **snr_db** (*float, optional*): [Default=30.0] Scalar value that controls assumed signal-to-noise ratio of the data in dB. Ignored if sigma_y is not None.
        - **weights** (*list[ndarray], optional*): [Default=None] List of 3-D weights array with same shape as sino.
        - **weight_type** (*string, optional*): [Default='unweighted'] Type of noise model used for data. If the ``weights`` array is not supplied, then the function ``cone3D.calc_weights`` is used to set weights using specified ``weight_type`` parameter.

                - Option "unweighted" corresponds to unweighted reconstruction;
                - Option "transmission" is the correct weighting for transmission CT with constant dosage;
                - Option "transmission_root" is commonly used with transmission CT data to improve image homogeneity;
                - Option "emission" is appropriate for emission CT data.
        - **positivity** (*bool, optional*): [Default=True] Boolean value that determines if positivity constraint is enforced. The positivity parameter defaults to True; however, it should be changed to False when used in applications that can generate negative image values.
        - **p** (*float, optional*): [Default=1.2] Scalar value in range :math:`[1,2]` that specifies the qGGMRF shape parameter.
        - **q** (*float, optional*): [Default=2.0] Scalar value in range :math:`[p,1]` that specifies the qGGMRF shape parameter.
        - **T** (*float, optional*): [Default=1.0] Scalar value :math:`>0` that specifies the qGGMRF threshold parameter.
        - **num_neighbors** (*int, optional*): [Default=6] Possible values are {26,18,6}. Number of neightbors in the qggmrf neighborhood. Higher number of neighbors result in a better regularization but a slower reconstruction.
        - **sharpness** (*float, optional*): [Default=0.0] Scalar value that controls level of sharpness in the reconstruction. ``sharpness=0.0`` is neutral; ``sharpness>0`` increases sharpness; ``sharpness<0`` reduces sharpness. Ignored in qGGMRF reconstruction if ``sigma_x`` is not None, or in proximal map estimation if ``sigma_p`` is not None.
        - **sigma_x** (*ndarray, optional*): [Default=None] Scalar value :math:`>0` that specifies the qGGMRF scale parameter. If None, automatically set with auto_sigma_x for each time point. Regularization should be controled with the ``sharpness`` parameter, but ``sigma_x`` can be set directly by expert users.
        - **sigma_p** (*float, optional*): [Default=None] Scalar value :math:`>0` that specifies the proximal map parameter. If None, automatically set with auto_sigma_p. Regularization should be controled with the ``sharpness`` parameter, but ``sigma_p`` can be set directly by expert users.
        - **max_iterations** (*int, optional*): [Default=3] Integer valued specifying the maximum number of iterations for proximal map estimation.
        - **stop_threshold** (*float, optional*): [Default=0.02] Scalar valued stopping threshold in percent. If stop_threshold=0.0, then run max iterations.
        - **num_threads** (*int, optional*): [Default=None] Number of compute threads requested when executed. If None, num_threads is set to the number of cores in the system
        - **NHICD** (*bool, optional*): [Default=False] If true, uses Non-homogeneous ICD updates
        - **verbose** (*int, optional*): [Default=1] Possible values are {0,1,2}, where 0 is quiet, 1 prints MACE reconstruction progress information, and 2 prints the MACE reconstruction as well as qGGMRF/proximal-map reconstruction and multinode computation information.
        - **lib_path** (*str, optional*): [Default=~/.cache/mbircone] Path to directory containing library of forward projection matrices.

    Returns:
        4-D numpy array: 4-D reconstruction with shape (num_time_points, num_img_slices, num_img_rows, num_img_cols) in units of :math:`ALU^{-1}`.
    """    
    
    if verbose: 
        print("initializing MACE...")
    # verbosity level for qGGMRF recon
    qGGMRF_verbose = max(0,verbose-1)     
    
    # agent weight
    if isinstance(prior_weight, (list, tuple, np.ndarray)):
        assert (len(prior_weight)==3), 'Incorrect dimensionality of prior_weight array.'
        beta = [1-sum(prior_weight)]
        for w in prior_weight:
            beta.append(w)
    else:
        beta = [1.-prior_weight,prior_weight/3.,prior_weight/3.,prior_weight/3.]
    assert(all(w>=0 for w in beta) and (sum(beta)-1.)<1e-5), 'Incorrect value of prior_weight given. All elements in prior_weight should be non-negative, and sum should be no greater than 1.'   
    
    # make denoiser_args an instance if necessary
    if not isinstance(denoiser_args, tuple):
        denoiser_args = (denoiser_args,) 

    # get sino shape 
    (Nt, num_views, num_det_rows, num_det_channels) = np.shape(sino)
    # if angles is a 1D array, form the 2D angles array s.t. same set of angles are used at every time point.
    if np.ndim(angles) == 1:
        angles = [angles for _ in range(Nt)]
    # Calculate automatic value of sinogram weights
    if weights is None:
        weights = cone3D.calc_weights(sino,weight_type)
    # Calculate automatic value of delta_pixel_image
    if delta_pixel_image is None:
        delta_pixel_image = delta_pixel_detector/magnification
    # Calculate automatic value of sigma_y
    if sigma_y is None:
        sigma_y = cone3D.auto_sigma_y(sino, weights, snr_db=snr_db, delta_pixel_image=delta_pixel_image, delta_pixel_detector=delta_pixel_detector)
    # Calculate automatic value of sigma_p
    if sigma_p is None:
        sigma_p = cone3D.auto_sigma_p(sino, delta_pixel_detector=delta_pixel_detector, sharpness=sharpness)
    # Fixed args dictionary used for multi-node parallelization
    constant_args = {'dist_source_detector':dist_source_detector, 'magnification':magnification,
                     'channel_offset':channel_offset, 'row_offset':row_offset, 'rotation_offset':rotation_offset,
                     'delta_pixel_detector':delta_pixel_detector, 'delta_pixel_image':delta_pixel_image, 'ror_radius':ror_radius,
                     'sigma_y':sigma_y, 'sigma_p':sigma_p, 'sigma_x':sigma_x,
                     'positivity':positivity, 'p':p, 'q':q, 'T':T, 'num_neighbors':num_neighbors,
                     'max_iterations':20, 'stop_threshold':stop_threshold,
                     'num_threads':num_threads, 'NHICD':NHICD, 'verbose':qGGMRF_verbose, 'lib_path':lib_path
    }
    # List of variable args dictionaries used for multi-node parallelization
    variable_args_list = [{'sino': sino[t], 'angles':angles[t], 'weights':weights[t]} 
                          for t in range(Nt)]
    # if init_image is not provided, use qGGMRF recon result as init_image.    
    if init_image is None:
        if verbose:
            start = time.time()
            print("Computing qGGMRF reconstruction at all time points. This will be used as MACE initialization point.") 
        
        if not (cluster_ticket is None):
            init_image = np.array(multinode.scatter_gather(cluster_ticket, cone3D.recon, 
                                                           variable_args_list=variable_args_list,
                                                           constant_args=constant_args,
                                                           verbose=qGGMRF_verbose))
        else:
            init_image = np.array([cone3D.recon(sino[t], angles[t], dist_source_detector, magnification,
                                                channel_offset=channel_offset, row_offset=row_offset, rotation_offset=rotation_offset,
                                                delta_pixel_detector=delta_pixel_detector, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius,
                                                weights=weights[t], sigma_y=sigma_y, sigma_x=sigma_x,
                                                positivity=positivity, p=p, q=q, T=T, num_neighbors=num_neighbors,
                                                max_iterations=20, stop_threshold=stop_threshold,
                                                num_threads=num_threads, NHICD=NHICD, verbose=qGGMRF_verbose, lib_path=lib_path) for t in range(Nt)])
        if verbose:
            end = time.time()
            elapsed_t = end-start
            print(f"Done computing qGGMRF reconstruction. Elapsed time: {elapsed_t:.2f} sec.")
            if not (save_path is None): 
                print("Save qGGMRF reconstruction to disk.")
                np.save(os.path.join(save_path, 'recon_qGGMRF.npy'), init_image) 
    
    if np.isscalar(init_image):
        [Nz,Nx,Ny], _ = cone3D.compute_img_size(num_views, num_det_rows, num_det_channels, 
                                                    dist_source_detector, magnification, 
                                                    channel_offset=channel_offset, row_offset=row_offset, rotation_offset=rotation_offset, 
                                                    delta_pixel_detector=delta_pixel_detector, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius)
        init_image = np.zeros((Nt, Nz, Nx, Ny)) + init_image 
    elif np.ndim(init_image) == 3:
        init_image = np.array([init_image for _ in range(Nt)])
    else:
        [_,Nz,Nx,Ny] = np.shape(init_image)
    
    # number of agents = image dimensionality.
    W = [np.copy(init_image) for _ in range(4)]
    X = [np.copy(init_image) for _ in range(4)]

    ######################## begin ADMM iterations ########################
    if verbose:
        print("Begin MACE ADMM iterations:")
    for itr in range(max_admm_itr):
        if verbose:
            print(f"Begin MACE iteration {itr}/{max_admm_itr}:")
            itr_start = time.time()
        # Modify constant_args and variable args respectively for proximal map estimation.
        constant_args['max_iterations'] = max_iterations
        for t in range(Nt):
            variable_args_list[t]['init_image'] = X[0][t]
            variable_args_list[t]['prox_image'] = W[0][t]
        # forward model prox map agent
        if not (cluster_ticket is None):
            X[0] = np.array(multinode.scatter_gather(cluster_ticket, cone3D.recon,
                                                      variable_args_list=variable_args_list,
                                                      constant_args=constant_args,
                                                      verbose=qGGMRF_verbose))
        else:
            X[0] = np.array([cone3D.recon(sino[t], angles[t], dist_source_detector, magnification,
                                          channel_offset=channel_offset, row_offset=row_offset, rotation_offset=rotation_offset,
                                          delta_pixel_detector=delta_pixel_detector, delta_pixel_image=delta_pixel_image, ror_radius=ror_radius,
                                          init_image=X[0][t], prox_image=W[0][t],
                                          weights=weights, sigma_y=sigma_y, sigma_p=sigma_p,
                                          positivity=positivity,
                                          max_iterations=max_iterations, stop_threshold=stop_threshold,
                                          num_threads=num_threads, NHICD=NHICD, verbose=qGGMRF_verbose, lib_path=lib_path) for t in range(Nt)])
        if verbose:
            print("Done forward model proximal map estimation.")
        # prior model denoiser agents
        # denoising in XY plane (along Z-axis)
        denoise_start = time.time()
        X[1] = denoiser_wrapper(W[1], denoiser, denoiser_args, permute_vector=(1,0,2,3), positivity=positivity) # shape should be after permutation (Nz,Nt,Nx,Ny)
        # denoising in YZ plane (along X-axis)
        X[2] = denoiser_wrapper(W[2], denoiser, denoiser_args, permute_vector=(2,0,1,3), positivity=positivity) # shape should be after permutation (Nx,Nt,Nz,Ny)
        # denoising in XZ plane (along Y-axis)
        X[3] = denoiser_wrapper(W[3], denoiser, denoiser_args, permute_vector=(3,0,1,2), positivity=positivity) # shape should be after permutation (Ny,Nt,Nz,Nx) 
        denoise_end = time.time()
        if verbose:
            denoise_elapsed = denoise_end - denoise_start
            print(f"Done denoising in all hyper-planes, elapsed time {denoise_elapsed:.2f} sec")
        # save X and W as npy files
        if not (save_path is None):
            for i in range(4):
                np.save(os.path.join(save_path, f'X{i}_itr{itr}.npy'), X[i])
                np.save(os.path.join(save_path, f'W{i}_itr{itr}.npy'), W[i])

        Z = sum([beta[k]*(2*X[k]-W[k]) for k in range(4)])
        for k in range(4):
            W[k] += 2*rho*(Z-X[k])
        if verbose:
            itr_end = time.time()
            itr_elapsed = itr_end-itr_start
            print(f"Done MACE iteration {itr}/{max_admm_itr}. Elapsed time: {itr_elapsed:.2f} sec.")
    ######################## end ADMM iterations ########################
    print("Done MACE reconstruction!")
    recon = sum([beta[k]*X[k] for k in range(4)])
    return recon
