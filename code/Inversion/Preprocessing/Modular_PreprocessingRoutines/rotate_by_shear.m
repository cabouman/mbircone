function img2 = rotate_by_shear(img, angle_degree, windowLen, interp_method  )

% rotate image by shearing 3 times
% https://www.ocf.berkeley.edu/~fricke/projects/israel/paeth/rotation_by_shearing.html
% shear image by assuming nyquist sampling
% shift ammount in samples or fraction of a sample

alpha = - tand(angle_degree/2);
beta = sind(angle_degree);
gamma = - tand(angle_degree/2);

img2 = shear_img_x(img, alpha, windowLen, interp_method);
img2 = shear_img_y(img2, beta, windowLen, interp_method);
img2 = shear_img_x(img2, gamma, windowLen, interp_method);

return
