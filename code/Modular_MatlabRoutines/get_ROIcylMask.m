function mask = get_ROIcylMask(sizeVect, startVect, stopVect);

mask = zeros(sizeVect);

cent = round( 0.5*[startVect(1) startVect(2)] + 0.5*[stopVect(1) stopVect(2)] );
radius = round( (stopVect(1)-startVect(1))/2 );
radiusList = zeros(1,sizeVect(3));
radiusList(startVect(3):stopVect(3)) = radius;

mask = mask_cylinder3D(mask, cent, radiusList);

return