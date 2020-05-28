function img2 = shift_img(img, shift_y, shift_x  )

% shift image by assuming nyquist sampling
% shift ammount in samples or fraction of a sample

N_x = size(img,2);
N_y = size(img,1);
N_z = size(img,3);

for k=1:N_z
	for i=1:N_y
		% shift each row in x
		img2(i,:,k) = signal_shift( img(i,:,k), shift_x, 0, 'cubicSpline' );
	end
		
	for j=1:N_x
		% shift each column in y
		img2(:,j,k) = signal_shift( img2(:,j,k), shift_y, 0, 'cubicSpline' );
	end
end

return
