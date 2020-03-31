function [ ] = generateColorbarEnvironment( folderName, dispName, img, lo, hi, figureNumber, opts)



outFolder = [folderName, '/', dispName, '_colorbar', '/'];

mkdir(outFolder);

figure(figureNumber); close; fig = figure(figureNumber);



for i = 1:size(img, 3)

    if(i<=size(img,3))

        if(lo ~= hi)
            imagesc(img(:,:,i), [lo,hi]);
        else % lo = hi
            if(lo>0)
                imagesc(img(:,:,i), [0, lo]);
            elseif(lo<0)
                imagesc(img(:,:,i), [lo, 0]);
            else % lo = hi = 0
                imagesc(img(:,:,i), [-1, 1]);
            end
        end
        
        

        colormap(gray(255));
        colorbar
        axis('image');
        xlabel([num2str(size(img,2)), ' pixels'], 'interpreter', 'latex');
        ylabel([num2str(size(img,1)), ' pixels'], 'interpreter', 'latex');
        title(['\verb|', dispName, '|', ' (Frame ', num2str(i), ' of ' num2str(size(img,3)), ')'], 'interpreter', 'latex');
        set(gca, 'fontsize', get(gca, 'fontsize')*opts.relFontSize);

        truesize(fig, size(img(:,:,i))*opts.figureDispSize)           % Adjusts figure s.t. the image has native resolution
        fig.Units = 'pixels';   % Ensures that resolution is 72 DPI (mac)
        resFac = opts.figurePrintSize/opts.figureDispSize;             % Factor by which saved image is higher resolution
        dpi = resFac*72*1;
        print(fig, [outFolder, dispName, '_Colorbar_', num2str(i,'%04.f'), '.png'], '-dpng', ['-r', num2str(dpi)]);
    end
end
end

