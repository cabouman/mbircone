function [ dataSetInfo ] = readDataSetPath_NSI( masterFile, plainParamsFile )

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'dataSetInfo';
value = '';
resolveFlag = '';


dataSetInfo.main_folderPath = plainParams(executablePath, get_set, masterFile, masterField, 'main_folderPath', value, resolveFlag) ;

relpath.objectScan_folderPath = plainParams(executablePath, get_set, masterFile, masterField, 'objectScan_folderPath', value, resolveFlag) ;
relpath.darkScanMean_filePath = plainParams(executablePath, get_set, masterFile, masterField, 'darkScanMean_filePath', value, resolveFlag) ;
relpath.darkScanStd_filePath = plainParams(executablePath, get_set, masterFile, masterField, 'darkScanStd_filePath', value, resolveFlag) ;
relpath.blankScanMean_filePath = plainParams(executablePath, get_set, masterFile, masterField, 'blankScanMean_filePath', value, resolveFlag) ;

relpath.corrections_filePath = plainParams(executablePath, get_set, masterFile, masterField, 'corrections_filePath', value, resolveFlag) ;
relpath.defects_filePath = plainParams(executablePath, get_set, masterFile, masterField, 'defects_filePath', value, resolveFlag) ;
relpath.driftReferenceStart_filePath = plainParams(executablePath, get_set, masterFile, masterField, 'driftReferenceStart_filePath', value, resolveFlag) ;
relpath.driftReferenceEnd_filePath = plainParams(executablePath, get_set, masterFile, masterField, 'driftReferenceEnd_filePath', value, resolveFlag) ;
relpath.occlusion_folderPath = plainParams(executablePath, get_set, masterFile, masterField, 'occlusion_folderPath', value, resolveFlag) ;


dataSetInfo.u_s = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'u_s', value, resolveFlag));
dataSetInfo.u_d1 = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'u_d1', value, resolveFlag));
dataSetInfo.v_d1 = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'v_d1', value, resolveFlag));
dataSetInfo.w_d1 = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'w_d1', value, resolveFlag));
dataSetInfo.Delta_dv = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'Delta_dv', value, resolveFlag));
dataSetInfo.Delta_dw = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'Delta_dw', value, resolveFlag));
dataSetInfo.N_dv = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'N_dv', value, resolveFlag));
dataSetInfo.N_dw = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'N_dw', value, resolveFlag));
dataSetInfo.numAcquiredScans = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'numAcquiredScans', value, resolveFlag));
dataSetInfo.TotalAngle = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'TotalAngle', value, resolveFlag));


dataSetInfo.flip_v_d0 = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'flip_v_d0', value, resolveFlag));
dataSetInfo.flip_w_d0 = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'flip_w_d0', value, resolveFlag));
dataSetInfo.do_mirrorScans_Horz = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'do_mirrorScans_Horz', value, resolveFlag));
dataSetInfo.do_mirrorScans_Vert = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'do_mirrorScans_Vert', value, resolveFlag));
dataSetInfo.flipParams_v_d1_w_d1 = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'flipParams_v_d1_w_d1', value, resolveFlag));
dataSetInfo.timesRotateScans90 = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'timesRotateScans90', value, resolveFlag));
dataSetInfo.cropLen = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'cropLen', value, resolveFlag));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convert relpath
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataSetInfo.objectScan_folderPath = getAbsPath_withNull(dataSetInfo.main_folderPath, relpath.objectScan_folderPath);
dataSetInfo.darkScanMean_filePath = getAbsPath_withNull(dataSetInfo.main_folderPath, relpath.darkScanMean_filePath);
dataSetInfo.darkScanStd_filePath = getAbsPath_withNull(dataSetInfo.main_folderPath, relpath.darkScanStd_filePath);
dataSetInfo.blankScanMean_filePath = getAbsPath_withNull(dataSetInfo.main_folderPath, relpath.blankScanMean_filePath);
dataSetInfo.corrections_filePath = getAbsPath_withNull(dataSetInfo.main_folderPath, relpath.corrections_filePath);
dataSetInfo.defects_filePath = getAbsPath_withNull(dataSetInfo.main_folderPath, relpath.defects_filePath);
dataSetInfo.driftReferenceStart_filePath = getAbsPath_withNull(dataSetInfo.main_folderPath, relpath.driftReferenceStart_filePath);
dataSetInfo.driftReferenceEnd_filePath = getAbsPath_withNull(dataSetInfo.main_folderPath, relpath.driftReferenceEnd_filePath);
dataSetInfo.occlusion_folderPath = getAbsPath_withNull(dataSetInfo.main_folderPath, relpath.occlusion_folderPath);


return

