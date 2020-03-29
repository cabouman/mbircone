mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../Modular_MatlabRoutines'));
addpath(fullfile(mfilepath,'MatlabRoutines'));

%% User input section


fName = '/Volumes/ORANGE4TB_DataSets/Binaries/CBMBIR_workfolder/FBP/NG8GE30101A_StepScan_2500Views.recon';

dataType = 'float32';
x = read3D(fName, dataType);


sliceID = floor(size(x,1)/2);
%sliceID = 48;  % comment if undisired
lo = 0;         % dark   will be mapped to this
hi = 1;         % bright will be mapped to this


%% Selection of dark and bright patches for normalization

slice = squeeze(x(sliceID, :,:));

f = figure(1);
clf
imagesc(slice);
truesize(f, size(slice));
colormap(gray)

disp('Select dark region');
[~, rect] = imcrop;
rect = round(rect);
patch = x(sliceID, rect(2):rect(2)+rect(4)-1, rect(1):rect(1)+rect(3)-1);
dark = mean(patch(:));

disp('Select bright region');
[~, rect] = imcrop;
rect = round(rect);
patch = x(sliceID, rect(2):rect(2)+rect(4)-1, rect(1):rect(1)+rect(3)-1);
bright = mean(patch(:));


%% Normalization
% get into [0,1]
x_new = x;
x_new = x_new - dark;
x_new = x_new / (bright-dark);

% get into [lo,hi]
x_new = x_new * (hi-lo);
x_new = x_new + lo;

%% storing

[a,b,c] = fileparts(fName);

fName_new = [a, '/', b, '.normalized', c];

write3D(fName_new, x_new, dataType);



