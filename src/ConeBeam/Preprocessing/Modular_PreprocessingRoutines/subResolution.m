function subResImg = subResolution(img, factor1, factor2, mode)
% This function artifically reduces resolution of an image by the
% user-defined factors.
% factors should be positive integers less than the size of the image

if(not(...
        strcmp(mode,'mean') || strcmp(mode,'sum')...
       )...
  )
    error('Error: mode must be ''mean'' or ''sum''');
end


[size1, size2] = size(img);
newSize1 = floor(size1/factor1);
newSize2 = floor(size2/factor2);

subResImg = zeros(newSize1, newSize2);

for i = 1:factor1
    for j = 1:factor2
        subResImg = subResImg + double(img(i:factor1:i+((newSize1-1)*factor1), j:factor2:j+((newSize2-1)*factor2)));
    end
end
if(strcmp(mode,'mean'))
    subResImg = subResImg/(factor1*factor2);
end


end

