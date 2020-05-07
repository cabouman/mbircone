function sino = photoncount2sino(scan, dark, blank)

% v = (l - d) / (b - d)
% y = - log v

sino = 0*scan;

v = normalize_scan(scan, dark, blank);
    
sino(v>0) = - log(v(v>0));
sino(v<=0) = 0;

sino(isnan(sino)) = 0;