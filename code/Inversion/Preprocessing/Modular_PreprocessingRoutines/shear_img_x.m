function img2 = shear_img_x(img, shearVal, windowLen, interp_method )

% shear image by assuming nyquist sampling
% shift ammount in samples or fraction of a sample

N_x = size(img,2);
N_y = size(img,1);

center_x = round(N_x/2);
center_y = round(N_y/2);

img2 = img;

for i=1:N_y
    y_val = -(i-center_y);
    shift_val = shearVal * y_val;

    % shift each row in x
	% img2(i,:) = signal_shift( img(i,:), shift_val, 0, 'fullsinc' );
    img2(i,:) = signal_shift( img(i,:), shift_val, windowLen, interp_method );

end

return
