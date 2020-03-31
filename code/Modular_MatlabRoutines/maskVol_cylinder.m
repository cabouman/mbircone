function [vol_new, x_m] = maskVol_cylinder(vol, truncRatios)

truncRadius = round( size(vol) .* truncRatios /2 );
center = round( size(vol)/2 );

cylinder_center = center(1:2);
cylinder_radiuslist = truncRadius(1)*ones(size(vol,3));

x_m = mask_cylinder3D(vol, cylinder_center, cylinder_radiuslist);
x_m(:,:,1:truncRadius(3)) = 0;
x_m(:,:,end-truncRadius(3)+1:end) = 0;

vol_new = vol;
vol_new(x_m==0) = 0;

return