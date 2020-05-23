function [ scans ] = rotate_scans( scans, num90 )

scans.darkmeanImg = rot90( scans.darkmeanImg, num90);
scans.darkvarImg = rot90( scans.darkvarImg, num90);
scans.blankmeanImg = rot90( scans.blankmeanImg, num90);
scans.blankvarImg = rot90( scans.blankvarImg, num90);
scans.driftReferenceStart_scan = rot90( scans.driftReferenceStart_scan, num90);
scans.driftReferenceEnd_scan = rot90( scans.driftReferenceEnd_scan, num90);

scans.defectivePixelMap = rot90( scans.defectivePixelMap, num90);

temp_obj = rot90( scans.object(:,:,1), num90);
new_object = zeros( size(temp_obj,1), size(temp_obj,2), size(scans.object,3) );
new_jig_scan = zeros( size(temp_obj,1), size(temp_obj,2), size(scans.jig_scan,3) );

for i=1:size(scans.object,3);
	new_object(:,:,i) = rot90( scans.object(:,:,i), num90);
end
for i=1:size(scans.jig_scan,3);
	new_jig_scan(:,:,i) = rot90( scans.jig_scan(:,:,i), num90);
end

scans.object = new_object;
scans.jig_scan = new_jig_scan;

end
