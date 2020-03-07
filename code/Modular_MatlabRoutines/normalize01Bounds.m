function [ out_img ] = normalize01Bounds( img, lo, hi )

if( lo>hi )
    error('ERROR in normalize01Bounds: lo>hi');
end

if( hi>lo )
    out_img = (img - lo) / (hi - lo);
else % hi=lo
    
    out_img = ones(size(img))*sign(lo)*0.5 + 0.5;
end


out_img(out_img>1) = 1;
out_img(out_img<0) = 0;

end

