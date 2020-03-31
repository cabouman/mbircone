function scans = read_scan_data(dataSetInfo, indexList)


scans.object = read_ObjScans(dataSetInfo.objectScan_folderPath, indexList);

if isempty(dataSetInfo.darkScanMean_filePath)
	scans.darkmeanImg = 0*scans.object(:,:,1);
else
	scans.darkmeanImg = read_scan( dataSetInfo.darkScanMean_filePath );
end

scans.blankmeanImg = read_scan( dataSetInfo.blankScanMean_filePath );
if isempty(dataSetInfo.darkScanStd_filePath)
	scans.darkvarImg = 0 * scans.blankmeanImg;
else
	sigma_noise = importdata(dataSetInfo.darkScanStd_filePath);
	scans.darkvarImg = sigma_noise.^2;
end

scans.blankvarImg = 0*scans.darkvarImg; % No variance for blank img


if isempty(dataSetInfo.defects_filePath)
	scans.defectivePixelMap = 0*scans.object(:,:,1) ;
else
	defectivePixelList_zeroIndex = readDefectivePixelsList( dataSetInfo.defects_filePath );
	defectivePixelList_oneIndex = defectivePixelList_zeroIndex + 1;
	scans.defectivePixelMap = compute_defectivePixelMap( defectivePixelList_oneIndex, 0*scans.object(:,:,1) );
end

if isempty(dataSetInfo.driftReferenceStart_filePath)
	scans.driftReferenceStart_scan = 0*scans.object(:,:,1) ;
else
	scans.driftReferenceStart_scan = read_scan( dataSetInfo.driftReferenceStart_filePath );
end

if isempty(dataSetInfo.driftReferenceEnd_filePath)
	scans.driftReferenceEnd_scan = 0*scans.object(:,:,1) ;
else
	scans.driftReferenceEnd_scan = read_scan( dataSetInfo.driftReferenceEnd_filePath );
end

if isempty(dataSetInfo.occlusion_folderPath)
	scans.jig_scan = 0*scans.blankmeanImg;
	% scans.jig_scan = 0*scans.object;
else
	scans.jig_scan = read_ObjScans(dataSetInfo.occlusion_folderPath, indexList);
end

