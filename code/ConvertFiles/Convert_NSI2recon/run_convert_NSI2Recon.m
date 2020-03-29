
fName_recon = '/Volumes/Data/Cone_Beam_Dataset/Lily_data/3D_scans/reverseengineer_device_scan20191104145822/recons/02_BHC/MTB-074_Ovation Whole Device Scan_105.34um 065BH.recon';
fName_head = '/Volumes/Data/Cone_Beam_Dataset/Lily_data/3D_scans/reverseengineer_device_scan20191104145822/recons/02_BHC/MTB-074_Ovation Whole Device Scan_105.34um 065BH.nsihdr';
% fName_body = '/Volumes/Data/Cone_Beam_Dataset/Lily_data/3D_scans/reverseengineer_device_scan20191104145822/recons/01_normal/MTB-074_Ovation Whole Device Scan_105.34um 0BH0000.nsidat';

x = convert_NSI2Recon_multiFile( fName_head, fName_recon );

% x = convert_NSI2Recon( fName_body, fName_head, fName_recon );
