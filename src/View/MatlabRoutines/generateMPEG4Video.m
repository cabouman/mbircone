function [ ] = generateMPEG4Video( folderName, dispName, x_norm )


vw = VideoWriter([folderName, '/', dispName], 'MPEG-4');
vw.FrameRate = 1;

vw.Quality = 100;
open(vw);
for i = 1:size(x_norm,3)
    writeVideo(vw, x_norm(:,:,i));
end
close(vw);


end

