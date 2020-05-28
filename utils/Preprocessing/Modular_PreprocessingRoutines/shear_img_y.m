function img2 = shear_img_y(img, shearVal, windowLen, interp_method )

% shear image by assuming nyquist sampling
% shift ammount in samples or fraction of a sample

N_x = size(img,2);
N_y = size(img,1);

center_x = round(N_x/2);
center_y = round(N_y/2);

img2 = img;

for j=1:N_x
    x_val = (j-center_x);
    shift_val = -shearVal * x_val;

    % shift each column in y
	% img2(:,j) = signal_shift( img(:,j), shift_val, 0, 'fullsinc' );
	img2(:,j) = signal_shift( img(:,j), shift_val, windowLen, interp_method );
    
end

return
