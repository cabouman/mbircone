function [ scans, parPCA, viewIndexList ] = readAllPCAData( dataSetInfo, preprocessingParams )


rawPCAData_object = readPCAFile( dataSetInfo.pcaObject );

viewIndexList = computeIndexList(preprocessingParams, rawPCAData_object.NumberImages, rawPCAData_object.RotationSector);



scans.object = readImagesInIndexList( rawPCAData_object.TIFFPathNames, viewIndexList );
scans.driftReferenceStart_scan = readImagesInIndexList( rawPCAData_object.TIFFPathNames, 0 );
scans.driftReferenceEnd_scan = readImagesInIndexList( rawPCAData_object.TIFFPathNames, rawPCAData_object.NumberImages );


scans.object = scans.object / rawPCAData_object.FreeRay;
scans.driftReferenceStart_scan = scans.driftReferenceStart_scan / rawPCAData_object.FreeRay;
scans.driftReferenceEnd_scan = scans.driftReferenceEnd_scan / rawPCAData_object.FreeRay;



if( dataSetInfo.mode == 1)
	%% for using photon count scans

	rawPCAData_dark = readPCAFile( dataSetInfo.pcaDark );
	darkscans = readImagesInIndexList( rawPCAData_dark.TIFFPathNames, 1:preprocessingParams.N_avg );
	darkscans = darkscans / rawPCAData_dark.FreeRay;
	scans.darkmeanImg = mean(darkscans, 3);
	scans.darkvarImg = var(darkscans, [], 3);

	rawPCAData_blank = readPCAFile( dataSetInfo.pcaBlank );
	blankscans = readImagesInIndexList( rawPCAData_blank.TIFFPathNames, 1:preprocessingParams.N_avg );
	blankscans = blankscans / rawPCAData_blank.FreeRay;
	scans.blankmeanImg = mean(blankscans, 3);
	scans.blankvarImg = var(blankscans, [], 3);

elseif( dataSetInfo.mode == 0)
	%% for using the transmission scans
	scans.darkmeanImg = zeros(size(scans.object(:,:,1)));
	scans.darkvarImg = zeros(size(scans.object(:,:,1)));

	scans.blankmeanImg = ones(size(scans.object(:,:,1)));
	scans.blankvarImg = zeros(size(scans.object(:,:,1)));
else
	error('Invalid mode');
end

parPCA = rawPCAData_object



end

