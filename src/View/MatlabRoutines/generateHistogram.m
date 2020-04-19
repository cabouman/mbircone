function [ ] = generateHistogram( folderName, dispName, img, range, figureNumber, numBins, description)


figure(figureNumber); close; figure(figureNumber);
numBins = 200;
% myhist = histogram(img(:), numBins); % this may fail when values too
% small line below is more robust
myhist = histogram(img(:), linspace(min(img(:)), max(img(:)), numBins+1));

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

