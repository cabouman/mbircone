function [ scans ] = mirror_scans( scans, mirrorDirection )

scans.darkmeanImg = mirror_image(scans.darkmeanImg, mirrorDirection);
scans.darkvarImg = mirror_image(scans.darkvarImg, mirrorDirection);
scans.blankmeanImg = mirror_image(scans.blankmeanImg, mirrorDirection);
scans.blankvarImg = mirror_image(scans.blankvarImg, mirrorDirection);
scans.driftReferenceStart_scan = mirror_image(scans.driftReferenceStart_scan, mirrorDirection);
scans.driftReferenceEnd_scan = mirror_image(scans.driftReferenceEnd_scan, mirrorDirection);

scans.defectivePixelMap = mirror_image(scans.defectivePixelMap, mirrorDirection);

for i=1:size(scans.object,3)
	scans.object(:,:,i) = mirror_image(scans.object(:,:,i), mirrorDirection);
end
for i=1:size(scans.jig_scan,3)
	scans.jig_scan(:,:,i) = mirror_image(scans.jig_scan(:,:,i), mirrorDirection);
end


end
