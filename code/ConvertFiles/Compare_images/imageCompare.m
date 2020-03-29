function [profile_list, params] = imageCompare(params)

if ~isfield(params, 'saveFolder')
    params.saveFolder = '~/Desktop';
end
if ~isfield(params, 'fNames')
    params.fNames = {};
end
if ~isfield(params, 'imgNames')
    params.imgNames = {};
end
if ~isfield(params, 'fullName')
    params.fullName = 'Test';
end
if ~isfield(params, 'isCrop')
    params.isCrop = true;
end
if ~isfield(params, 'isNormalizeByFirst')
    params.isNormalizeByFirst = true;
end
if ~isfield(params, 'isreducedMargin')
    params.isreducedMargin = true;
end
if ~isfield(params, 'isProfile')
    params.isProfile = true;
end
if ~isfield(params, 'magnification')
    params.magnification = 1;
end
if ~isfield(params, 'colors')
    params.colors = {};
end
if ~isfield(params, 'lines')
    params.lines = {};
end
if ~isfield(params, 'LineWidth')
    params.LineWidth = 0.5;
end


% add extra param elements if not enough
for i=length(params.lines)+1:length(params.fNames)
    params.lines{i} = '-';
end
for i=length(params.colors)+1:length(params.fNames)
    params.colors{i} = 'r';
end
for i=length(params.imgNames)+1:length(params.fNames)
    [~, params.imgNames{i}] = fileparts(params.fNames{i});
end

% remove excess param elements
params.lines = params.lines(1:length(params.fNames));
params.colors = params.colors(1:length(params.fNames));
params.imgNames = params.imgNames(1:length(params.fNames));


saveFolder_imgs_tif = [params.saveFolder, '/', 'imgs_tif'];
saveFolder_imgs_png = [params.saveFolder, '/', 'imgs_png'];
saveFolder_line_png = [params.saveFolder, '/', 'line_png'];

createFolder_purge(saveFolder_imgs_tif);
createFolder_purge(saveFolder_imgs_png);
createFolder_purge(saveFolder_line_png);

img_ref = imread(params.fNames{1});
quantizedRange = double([intmin(class(img_ref)), intmax(class(img_ref))]);
img_ref = double(img_ref)/(quantizedRange(2));


f = figure(1);
imagesc(img_ref, [0,1]);
truesize(f, size(img_ref)*params.magnification);
colormap(gray(1000))

if(params.isCrop)
    [~, rect] = imcrop;
    rect = round(rect);
else
    rect = [1, 1, size(img_ref, 2), size(img_ref, 1)];
end
% disp(['rect = [', num2str(rect), ']'])
img_ref_crop = img_ref(rect(2):rect(2)+rect(4)-1, rect(1):rect(1)+rect(3)-1);

imagesc(img_ref_crop, [0,1]);
truesize(f, size(img_ref_crop)*params.magnification);
colormap(gray(1000))

if(params.isProfile)
    if ~isfield(params, 'xi') || ~isfield(params, 'yi')
        [~, ~, ~, params.xi, params.yi] = improfile;
    end
else
    params.xi = [];
    params.yi = [];
end

profile_list = {};
for i = 1:length(params.fNames)
    
    img = imread(params.fNames{i});
    img = double(img)/(quantizedRange(2));

    if params.isNormalizeByFirst
        disp('Normalizing');
        img = LS_fit_vol(img, img_ref);
    end

    img_crop = img(rect(2):rect(2)+rect(4)-1, rect(1):rect(1)+rect(3)-1);

    [cx,cy,profile_list{i}] = improfile(img_crop, params.xi, params.yi);
    
    f = figure(i);
    imagesc(img_crop, [0,1]);
    % imshow(img_crop, [0,1]);
    truesize(f, size(img_crop)*params.magnification);
    colormap(gray(1000))
    
    line(cx, cy, 'color', params.colors{i}, 'LineWidth', params.LineWidth);

    filename_imgs_tif = [saveFolder_imgs_tif, '/', params.imgNames{i}, '.tif'];
    filename_imgs_png = [saveFolder_imgs_png, '/', params.imgNames{i}, '.png'];
    filename_line_png = [saveFolder_line_png, '/', params.imgNames{i}, '.png'];
    
    if params.isreducedMargin
        set(gca,'position',[0 0 1 1],'units', 'normalized')
    else
        title(params.imgNames{i}, 'Interpreter', 'none');
    end
    
    if(params.isProfile)
        saveas(f, filename_line_png);
    end
    imwrite(img_crop, filename_imgs_tif);
    imwrite(img_crop, filename_imgs_png);
    
end
close all;

%%
if(params.isProfile)
    f = figure(1);
    clf
    for i = 1:length(params.fNames)
        hold on;
        plot(profile_list{i}, [params.colors{i},params.lines{i}]);
        hold off    
    end

    leg = legend(params.imgNames{:});
    set(leg, 'Interpreter', 'none');
    xlabel('Cross-section');
    ylabel('Intensity');

    filename_plot_png = [params.saveFolder, '/', char(strrep(params.fullName,' ','')), '.png'];
    filename_plot_fig = [params.saveFolder, '/', char(strrep(params.fullName,' ','')), '.fig'];
    saveas(f,filename_plot_png);
    saveas(f,filename_plot_fig);
end
close all;


return
