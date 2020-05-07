function [dark, bright] = get_dark_bright(slice)

%% Selection of dark and bright patches for normalization

f = figure;
clf
imagesc(slice);
truesize(f, size(slice));
colormap(gray);

disp('Select dark region');
[~, rect] = imcrop;
rect = round(rect);
patch = slice(rect(2):rect(2)+rect(4)-1, rect(1):rect(1)+rect(3)-1);
dark = mean(patch(:))

disp('Select bright region');
[~, rect] = imcrop;
rect = round(rect);
patch = slice(rect(2):rect(2)+rect(4)-1, rect(1):rect(1)+rect(3)-1);
bright = mean(patch(:))

close(f)

return
