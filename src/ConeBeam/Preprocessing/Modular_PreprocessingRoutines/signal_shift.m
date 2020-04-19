function y_out = signal_shift(x_in, shift, windowLen, interp_method)

switch interp_method
    case 'freqsinc'
        y_out = fshift(x_in, shift);

    case 'windowsinc'
        y_out = signal_interp(x_in, shift, windowLen, 'windowsinc');

    case 'fullsinc'
        y_out = signal_interp(x_in, shift, 0, 'fullsinc');

    case 'cubicSpline'
    	y_out = signal_interp(x_in, shift, 0, 'cubicSpline');

    case 'linear'
    	y_out = signal_interp(x_in, shift, 0, 'linear');

    case 'quadSpline'
        y_out = signal_interp(x_in, shift, 0, 'quadSpline');

    case 'quadSplineFast'
        y_out = signal_interp(x_in, shift, 0, 'quadSplineFast');    

    otherwise
        y_out = signal_interp(x_in, shift, 0, interp_method);      

end
        