function y_out = signal_interp(x_in, shift, windowLen, interp_method)

x_in = x_in(:);

pad_ammount = length(x_in) ;
x_padded = padarray( x_in, pad_ammount, 'symmetric' );
N = length(x_padded);

% seperate integer and fractional shift
shift_int = floor(abs(shift)) * sign(shift) ;
shift_frac = ( abs(shift) - floor(abs(shift)) ) * sign(shift) ;

switch interp_method
    case 'fullsinc'
        % use MATLAB's ful sinc interpolation
        x_shifted = delayseq(x_padded(:),shift);
        x_shifted = x_shifted(:);

     case 'windowsinc'  
        % shift by integer 
        x_shifted_int = delayseq(x_padded(:),shift_int);

        if shift_frac~=0
            % shift by fraction 

            h = sinc([-windowLen:windowLen]-shift_frac);
            h = h(:);
            h = h.*hamming(2*windowLen+1);

            temp = conv(x_shifted_int, h);
            x_shifted = temp(windowLen+1:end-windowLen);

            % x_shifted = conv(x_shifted_int, h, 'same');
        else
            x_shifted = x_shifted_int;
        end
        x_shifted = x_shifted(:);

    case 'linear'
        % use MATLAB's linear interpolation
        x_shifted = interp1(1:length(x_padded), x_padded(:), [1:length(x_padded)]-shift, 'linear');
        x_shifted = x_shifted(:);

    case 'cubicSpline'
        % use MATLAB's spline interpolation
        x_shifted = spline(1:length(x_padded), x_padded(:), [1:length(x_padded)]-shift);
        x_shifted = x_shifted(:);

    case 'quadSpline'
        % shift by integer 
        x_shifted_int = delayseq(x_padded(:),shift_int);
        t = 1:length(x_shifted_int);
        coeff_vect = qbanal(x_shifted_int);

        for i=1:length(x_shifted_int)
            x_shifted(i)=qbinterp(coeff_vect, t(i)-shift_frac) ;        
        end
        
    case 'quadSplineFast'
        % shift by integer 
        x_shifted_int = delayseq(x_padded(:),shift_int);
        t = 1:length(x_shifted_int);
        
        if exist('filiirs', 'file')~=3
            backupDir = pwd;
            mfilepath=fileparts(which(mfilename));
            cd(fullfile(mfilepath,'spline_interpolation'));
            mex COMPFLAGS='$COMPFLAGS -Wall' filiirs.c
            cd(backupDir);
        end
        

        % tic
        coeff_vect = fspline(x_shifted_int,2);
        % coeff_time = toc;
        % tic
        for i=1:length(x_shifted_int)
            x_shifted(i)=qbinterp(coeff_vect, t(i)-shift_frac) ;        
        end
        % interp_time = toc;

        % fprintf('Times: coeff=%d interp=%d \n',coeff_time,interp_time); 

    otherwise
         x_shifted = x_padded;

end


y_out = x_shifted(pad_ammount+1:end-pad_ammount);

return
