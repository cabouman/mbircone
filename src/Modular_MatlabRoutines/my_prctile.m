function y = my_prctile(x,p)
% y = my_prctile(x,p)
%
% Limited-use replacement of 'prctile' function in case the
% Statistics and Machine Learning toolbox isn't available.
% Output y will be the value corresponding to p percentile derived
% from the values in input x. 
%
% Should give equivalent result to 'prctile' function for the case
% that input x is a 1D array, and input p is a scalar.
%
% Input x can be multidimensional array, but will be collapsed to 1D.
%

x=x(:);
xs=sort(x);

% percentiles corresponding to values in sorted 'x': 100*([1:N]-0.5)/N
N=length(x);
idxf = p*N/100 + 0.5;	% index units

if(idxf <= 1)
   y=min(xs);
elseif(idxf >= N)
   y=max(xs);
else

   idx1=floor(idxf);
   idx2=idx1+1;

   % use linear interpolation between nearest x values, as in Matlab's prctile
   x1=xs(idx1);
   x2=xs(idx2);
   y = x1 + (x2-x1)*(idxf-idx1)/(idx2-idx1);
  
end

return

