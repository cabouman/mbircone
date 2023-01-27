import json
import os,sys
import numpy as np
import argparse
from configparser import ConfigParser
from mbircone import preprocess, cone3D
from demo_utils import plot_gif
import pprint
pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', dest='config_path', default='./params.yml')
args = parser.parse_args()
def main():
    config_object = ConfigParser()
    config_object.read(args.config_path)
    ########## preprocess parameters 
    preprocess_params = config_object["Preprocess"]
    nsi_config_file_path = preprocess_params["nsi_config_file_path"]
    obj_scan_path = preprocess_params["obj_scan_path"]
    blank_scan_path = preprocess_params["blank_scan_path"]
    dark_scan_path = preprocess_params["dark_scan_path"]
    crop_detector_factor = json.loads(preprocess_params["crop_detector_factor"])
    downsample_detector_factor = json.loads(preprocess_params["downsample_detector_factor"])
    subsample_view_factor = int(preprocess_params["subsample_view_factor"])

    ########## recon parameters
    recon_params = config_object["Recon"]    
    sharpness = float(recon_params["sharpness"])    
    weight_type = recon_params["weight_type"]    
    max_iterations = int(recon_params["max_iterations"])    
    lib_path = recon_params["lib_path"] 
    
    ########## postprocess parameters 
    postprocess_params = config_object["Postprocess"]
    output_dir = postprocess_params["output_dir"] 
    recon_file_name = postprocess_params["recon_file_name"] 
    vmin = float(postprocess_params["vmin"])
    vmax = float(postprocess_params["vmax"])
    os.makedirs(output_dir, exist_ok=True)
       

    print("\n*******************************************************",
          "\n***** Loading NSI scan images and geometry params *****",
          "\n*******************************************************")
    obj_scan, blank_scan, dark_scan, angles, geo_params = \
        preprocess.NSI_load_scans_and_params(nsi_config_file_path, obj_scan_path, blank_scan_path, dark_scan_path,
                                             downsample_factor=downsample_detector_factor, crop_factor=crop_detector_factor,
                                             subsample_view_factor=subsample_view_factor)
    print("MBIR geometry paramemters:")
    pp.pprint(geo_params)
    print('obj_scan shape = ', obj_scan.shape)
    print('blank_scan shape = ', blank_scan.shape)
    print('dark_scan shape = ', dark_scan.shape)
     
    # extract mbircone geometry params required for recon
    dist_source_detector = geo_params["dist_source_detector"]
    magnification = geo_params["magnification"]
    delta_det_row = geo_params["delta_det_row"]
    delta_det_channel = geo_params["delta_det_channel"]
    det_channel_offset = geo_params["det_channel_offset"]
    det_row_offset = geo_params["det_row_offset"]
     
    print("\n*******************************************************",
          "\n** Computing sino and sino weights from scan images ***",
          "\n*******************************************************")
    sino, weights = preprocess.transmission_CT_preprocess(obj_scan, blank_scan, dark_scan,
                                                          weight_type=weight_type)
    print('sino shape = ', sino.shape)
    
    print("\n*******************************************************",
          "\n*********** Performing MBIR reconstruction ************",
          "\n*******************************************************")
    recon_qGGMRF = cone3D.recon(sino, angles, dist_source_detector, magnification,
                                det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                delta_det_row=delta_det_row, delta_det_channel=delta_det_channel,
                                sharpness=sharpness, weights=weights,
                                max_iterations=max_iterations, verbose=1,
                                lib_path=lib_path)
    
    print("qGGMRF recon finished. recon shape = ", np.shape(recon_qGGMRF))


    print("\n*******************************************************",
          "\n********** Plotting sinogram images and gif ***********",
          "\n*******************************************************")
    np.save(os.path.join(output_dir, recon_file_name+'.npy'), recon_qGGMRF)
    plot_gif(recon_qGGMRF, output_dir, recon_file_name, vmin=vmin, vmax=vmax)


if __name__ == '__main__':
     main()
