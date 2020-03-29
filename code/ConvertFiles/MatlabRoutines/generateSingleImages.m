function [ ] = generateSingleImages( folderName, dispName, x_norm, frames_fileType)


singleImagesOutputFolder = [folderName, '/', dispName, '_', frames_fileType, '/'];

mkdir(singleImagesOutputFolder);

for j = 1:size(x_norm,3)
    fName_prefix = [singleImagesOutputFolder, dispName, sprintf('%04d', j)];
    if(strcmp(frames_fileType, 'jpg'))
        fName = [fName_prefix, '.jpg'];
        imwrite(x_norm(:,:,j), fName, 'JPEG', 'Quality', 100);
    elseif(strcmp(frames_fileType, 'tif'))
        fName = [fName_prefix, '.tif'];
        img = uint16(x_norm(:,:,j)*2^16-1);
        imwrite(img, fName);
    end
    
end

end

