function y_out = fshift(x_in, shift)

% Uses sinc interpolation in the frequency domain to shift signals by fractional ammount
% Uses symmetric padding
% works by applying a linear phase in the spectrum domain and is equivalent to CIRCSHIFT for 
% integer values of argument shift (to machine precision).

x_in = x_in(:);
pad_ammount = 2*ceil(abs(shift));
x_padded = padarray( x_in, pad_ammount, 'symmetric' );

N = length(x_padded);

r = floor(N/2)+1; 
f = ((1:N)-r)/(N/2); 
p = exp(j*shift*pi*f)'; 
y = ifft(fft(x_padded).*ifftshift(p)); 
y = real(y); 

y_out = y(pad_ammount+1:end-pad_ammount);

return
