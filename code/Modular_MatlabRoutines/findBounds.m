function [ lo_new, hi_new ] = findBounds( img, opts )

mode = opts.mode;
factor = opts.prctileSubsampleFactor;
if(startsWith_(mode, 'percentile'))
    mode = mode(length('percentile '):end);
    
    temp = sscanf(mode, '%f');
    loPerc = temp(1, :);
    hiPerc = temp(2, :);
    
    lo = prctile(img(1:factor:end), loPerc);
    hi = prctile(img(1:factor:end), hiPerc);
end

if(startsWith_(mode, 'absolute'))
    mode = mode(length('absolute '):end);
    
    temp = sscanf(mode, '%f');
    lo = temp(1, :);
    hi = temp(2, :);
    
end

%% mapping [0,1] to [target_lo, target_hi]
% Use [0,1] to get default output
target_lo = opts.target_lo;
target_hi = opts.target_hi;
lo_new = lo - (hi-lo) * ( (target_lo) /(target_hi-target_lo));
hi_new = hi + (hi-lo) * ((1-target_hi)/(target_hi-target_lo));



end

