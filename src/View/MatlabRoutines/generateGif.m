function [ ] = generateGif( folderName, dispName, img_norm )


gifFilename = [dispName, '.gif'];
writeGif(uint8(img_norm*255), [folderName, '/', gifFilename]);

end

