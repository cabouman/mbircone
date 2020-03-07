function imageComare(saveFolder, fNames, titles, titles_all, isProfile, xi, yi, colors, lines, isCrop, magnification, isreducedMargin)

x_full = imread(fNames{1});


range = double([intmin(class(x_full)), intmax(class(x_full))]);

x_full = double(x_full)/(range(2));

f = figure(1);
imagesc(x_full, [0,1]);
truesize(f, size(x_full)*magnification);
colormap(gray)

if(isCrop)
    [~, rect] = imcrop;
    rect = round(rect);
else
    rect = [1, 1, size(x_full, 2), size(x_full, 1)];
end

% disp(['rect = [', num2str(rect), ']'])

x_crop = x_full(rect(2):rect(2)+rect(4)-1, rect(1):rect(1)+rect(3)-1);

imagesc(x_crop, [0,1]);
truesize(f, size(x_crop)*magnification);
colormap(gray)

if(isProfile)
    [cx, cy, ~, xi, yi] = improfile;
else
    cx = [];
    cy = [];
    xi = [];
    yi = [];
end
% disp(['xi = [', num2str(xi'), ']'''])
% disp(['yi = [', num2str(yi'), ']'''])



c = {};
for i = 1:length(fNames)
    
    x_full = imread(fNames{i});
    x_full = double(x_full)/(range(2));
    x_crop = x_full(rect(2):rect(2)+rect(4)-1, rect(1):rect(1)+rect(3)-1);
    [~,~,c{i}] = improfile(x_crop, xi, yi);
    
    f = figure(i);
    % imagesc(x_crop, [0,1]);
    imshow(x_crop, [0,1]);
    truesize(f, size(x_crop)*magnification);
    colormap(gray)
    
    line(cx, cy, 'color', colors{i});
    
    if isreducedMargin
        set(gca,'position',[0 0 1 1],'units','normalized')
    else
        title(titles{i}, 'Interpreter', 'none');
    end
    filename = [saveFolder, '/', strrep(titles{i},' ',''), '.png'];
    
    saveas(f,filename);
    
end


%%
if(isProfile)
    f = figure(10);
    clf
    for i = 1:length(fNames)
        hold on;
        plot(c{i}, [colors{i},lines{i}]);
        hold off    
    end

    leg = legend(titles{:});
    set(leg, 'Interpreter', 'none');
    
    xlabel('Cross-section');
    ylabel('Intensity');

    filename = [saveFolder, '/', char(strrep(titles_all,' ','')), '.png'];
    filename_m = [saveFolder, '/', char(strrep(titles_all,' ','')), '.fig'];
    saveas(f,filename);
    saveas(f,filename_m);
end

end
