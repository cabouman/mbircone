import os,sys
import numpy as np
import urllib.request
import tarfile
from keras.models import model_from_json
import mbircone
import display_utils

if __name__ == '__main__':
    ################ Download and extract data for this demo
    # Download phantom and params files
    download_url = 'https://github.com/dyang37/mbircone_data/raw/master/demo_data.tar.gz'
    urllib.request.urlretrieve(download_url, 'temp.tar') 
    # Extract tarball file
    tar_file = tarfile.open('temp.tar')
    tar_file.extractall('./demo_data/')
    tar_file.close()
    os.remove('temp.tar') 
    
    ################ geometry params
    num_views = 75
    dist_source_detector = 839.0472
    magnification = 5.572128439964856
    delta_pixel_detector = 0.25
    num_det_rows = 28
    num_det_channels = 240
    
    ################ MACE recon params
    max_admm_itr = 10
    prior_weight = 0.5

    ################ obtain noisy sinogram from phantom image and noise level
    # 1. obtain view angles: assume angle range to be 2pi, and all view angles are uniformly spaced.
    angles = np.linspace(0, 2*np.pi, num_views, endpoint=False)

    # 2. obtain clean sinogram by forward projecting phantom
    phantom = np.load('./demo_data/phantom_3D.npy')
    sino = mbircone.cone3D.project(phantom,angles,
                num_det_rows, num_det_channels,
                dist_source_detector, magnification,
                delta_pixel_detector=delta_pixel_detector)
    
    # 3. calculate weights for both transmission noise and MBIR reconstruction
    weights = mbircone.cone3D.calc_weights(sino, weight_type='transmission')
    
    # 4. Add noise to clean sinogram
    noise_sigma = 0.01
    noise = noise_sigma * 1./np.sqrt(weights) * np.random.normal(size=(num_views,num_det_rows,num_det_channels))
    
    # 5. add noise to clean sinogram
    sino_noisy = sino + noise
    
    # output image and results directory
    output_dir = './output/mace3D/'
    os.makedirs(output_dir, exist_ok=True)
    
    # initialize denoiser function used in MACE reconstruction
    denoiser = mbircone.mace.keras_denoiser 
    # load denoiser model structure and weights
    json_path = './demo_data/dncnn_params/model_dncnn/model.json'
    weight_path = './demo_data/dncnn_params/model_dncnn/model.hdf5'
    json_file = open(json_path, 'r')
    denoiser_model_json = json_file.read()
    json_file.close()
    denoiser_model = model_from_json(denoiser_model_json)
    denoiser_model.load_weights(weight_path)
    
    ################ MACE reconstruction
    recon_mace = mbircone.mace.mace3D(sino_noisy, angles, dist_source_detector, magnification,
            denoiser=denoiser, denoiser_args=(denoiser_model),
            max_admm_itr=max_admm_itr, prior_weight=prior_weight,
            delta_pixel_detector=delta_pixel_detector,
            weight_type='transmission')
    
    ################ Reconstruction results post processing
    # save recon results as a numpy array
    np.save(os.path.join(output_dir,"recon_mace.npy"), recon_mace)
    # plot sinogram data
    display_utils.plot_image(sino_noisy[0],title='sinogram view 0, noise level=0.05',filename=os.path.join(output_dir,'sino_noisy.png'),vmin=0,vmax=4)
    display_utils.plot_image(sino[0],title='clean sinogram view 0',filename=os.path.join(output_dir,'sino_clean.png'),vmin=0,vmax=4)
    display_utils.plot_image(noise[0],title='sinogram additive Gaussian noise,  view 0',filename=os.path.join(output_dir,'sino_transmission_noise.png'),vmin=-0.08,vmax=0.08)
    # plot an axial slice of phantom and recon
    display_utils.plot_image(phantom[1],title='phantom, axial slice 1',filename=os.path.join(output_dir,'phantom_slice.png'),vmin=0,vmax=0.5)
    display_utils.plot_image(recon_mace[1],title='MACE reconstruction, axial slice 1',filename=os.path.join(output_dir,'recon_mace_slice.png'),vmin=0,vmax=0.5)
    # plot 3D phantom and recon image volumes as gif images.
    display_utils.plot_gif(phantom,output_dir,'phantom',vmin=0,vmax=0.5)
    display_utils.plot_gif(recon_mace,output_dir,'recon_mace',vmin=0,vmax=0.5)

