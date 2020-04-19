function [ out ] = cropScan( img, shifty0, shifty1, shiftx0, shiftx1 )

out = img(1+shifty0:end-shifty1, 1+shiftx0:end-shiftx1, :);


end

