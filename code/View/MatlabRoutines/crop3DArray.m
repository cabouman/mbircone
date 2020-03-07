function [ img_out, range1, range2, range3] = crop3DArray(img, range1, range2, range3)

%% countingNumbers1 = 1:size(img,1);
%% countingNumbers2 = 1:size(img,2);
%% countingNumbers3 = 1:size(img,3);
%% 
%% range1 = eval(['countingNumbers1(' range1 ')'])
%% range2 = eval(['countingNumbers2(' range2 ')'])
%% range3 = eval(['countingNumbers3(' range3 ')'])

range1 = interpretSymbolicRange(img, 1, range1);
range2 = interpretSymbolicRange(img, 2, range2);
range3 = interpretSymbolicRange(img, 3, range3);

img_out = img(range1, range2, range3);


end


