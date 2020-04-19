function [ outimg ] = normalize01( img )

range = (max(img(:)) - min(img(:)));
if( range > 0 )
    outimg = (img - min(img(:))) / range;
else
    outimg = img - min(img(:));
end

