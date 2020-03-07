function [ ] = generateHistogram( folderName, dispName, img, range, figureNumber, numBins, description)


figure(figureNumber); close; figure(figureNumber);
myhist = histogram(img(:), 200);
title(['Histogram of \verb|', dispName, '|', description], 'interpreter', 'latex');
ylabel('Frequency');
xlabel('Intensity Value');
set(gca,'yscale','log');
MIN = range(1);
MAX = range(2);
if(MAX>MIN)
    xlim(range);
end
saveas(myhist, [folderName, '/', dispName, description, '_histogram.jpeg']);

end

