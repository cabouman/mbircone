function writeGif( image, filePath )
% image needs to be uint8 i.e. range 0 to 255.

%Write the first frame to a file 
imwrite(image(:,:,1), filePath,'gif','writemode','overwrite',...
        'LoopCount',inf,'DelayTime',0);

%Loop through and write the rest of the frames
for i=2:size(image,3)
     imwrite(image(:,:,i), filePath,'gif','writemode','append','DelayTime',0)
end

end

