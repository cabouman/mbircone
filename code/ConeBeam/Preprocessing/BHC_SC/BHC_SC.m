%function [] = BHC+SC(masterFile, plainParamsFile)
clear all
%masterFile = '../../../../control/Inversion/QGGMRF/master.txt';
masterFile = '../../../../control/Inversion/QGGMRF/4D_replica_000000_master.txt';
plainParamsFile = '../../../plainParams/plainParams.sh';
mfilename = 'BHC+SC';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../Modular_PreprocessingRoutines'));
addpath(fullfile(mfilepath,'../../../Modular_MatlabRoutines'));

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Read File Names and such
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Read File Names ...');
binaryFNames = readBinaryFNames( masterFile, plainParamsFile );

preprocessingParams = readPreprocessingParams(masterFile, plainParamsFile);

printStruct(binaryFNames, 'binaryFNames');
printStruct(preprocessingParams, 'preprocessingParams');



%% ----------------- File Names ---------------------
DS = 20;

fName_x_s = prependAppendFileName( 'BHC/', binaryFNames.recon, '.x_s');
fName_x_s_small = prependAppendFileName( 'BHC_small/', binaryFNames.recon, '.x_s.small');

fName_Ax_s = prependAppendFileName( 'BHC/', binaryFNames.sino, '.Ax_s');
fName_Ax_s_small = prependAppendFileName( 'BHC_small/', binaryFNames.sino, '.Ax_s.small');

fName_w_small = prependAppendFileName( 'BHC_small/', binaryFNames.wght, '.mask.small');

fName_y_0_small = prependAppendFileName( 'BHC_small/', binaryFNames.sino, '.y_0.small');

fName_delta_small = prependAppendFileName( 'BHC_small/', binaryFNames.sino, '.delta.small');
fName_delta_blur_small = prependAppendFileName( 'BHC_small/', binaryFNames.sino, '.delta_blur.small');

fName_bhc_sino_small = prependAppendFileName( 'BHC_small/', binaryFNames.sino, '.bhc_sino.small');
fName_y_new_small = prependAppendFileName( 'BHC_small/', binaryFNames.sino, '.y_new.small');
fName_y_0_small = prependAppendFileName( 'BHC_small/', binaryFNames.sino, '.y_0.small');



[pathstr, ~, ~] = fileparts(fName_x_s);
fName_statFile = [pathstr, '/stats.mat'];


%% ----------------- Segmentation ---------------------



%% Segment x -> x_s
x = read3D(binaryFNames.recon, 'float32');


%% -- Fixed Threshold --
%Threshold = 0.075;
%x_s = x;
%x_s(x< Threshold) = -Threshold;
%x_s(x>=Threshold) = Threshold;
%x_s = x_s / (2*Threshold) + 1/2;

%% -- Adaptive Method --

x_s1 = zeros(size(x));
for i = 1:size(x,1)
    x_s1(i,:,:) = reshape(imbinarize(squeeze(x(i,:,:))), size(x(i,:,:)));
end

x_s2 = zeros(size(x));
for i = 1:size(x,2)
    x_s2(:,i,:) = reshape(imbinarize(squeeze(x(:,i,:))), size(x(:,i,:)));
end

x_s3 = zeros(size(x));
for i = 1:size(x,3)
    x_s3(:,:,i) = reshape(imbinarize(squeeze(x(:,:,i))), size(x(:,:,i)));
end

x_s = x_s1 .* x_s2 .* x_s3;

% --

write3D(fName_x_s_small, x_s(1:DS:end,:,:), 'float32');
write3D(fName_x_s, x_s, 'float32');
clear x_s x

%% ----------------- Forward projection ---------------------


%% Forward project x_s -> Ax_s
command = ['bash ../../../Inversion/run/forwardProject.sh -M ', masterFile, ' -i ', fName_x_s, ' -o ', fName_Ax_s];
[exit_status, commandLineOutput] = system(command);
if(exit_status==1)
    error('Error with system command');
end
Ax_s = read3D(fName_Ax_s, 'float32');
y_0 = read3D(binaryFNames.origSino, 'float32');


write3D(fName_Ax_s_small, Ax_s(:,:,1:DS:end), 'float32');
write3D(fName_y_0_small, y_0(:,:,1:DS:end), 'float32');



%% ----------------- Preparation of variables ---------------------


%% Powers of y_0
start_exponent = 0;
degree = 5;
exp_range = start_exponent : degree;




%% Blurred perfect sinogram
clear sigma_g clear power_p
sigma_g = [];
power_p =[];


len_gam_Y = length(exp_range);
len_gam_H = length(sigma_g);
len_gam = len_gam_Y + len_gam_H;

indicesY = 1:len_gam_Y;
indicesH = (1:len_gam_H) + len_gam_Y;

YH = zeros(length(Ax_s(:)), len_gam);



%% Prepare Y = powers of y_0
YH(:,indicesY) = repmat(y_0(:), 1, length(exp_range)) .^ repmat(exp_range, length(y_0(:)), 1);



%% Prepare H = blurred sino
for i_H = 1:length(indicesH)
    i_YH = indicesH(i_H);

    temp = imgaussfilt(Ax_s.^power_p(i_H), sigma_g(i_H), 'padding', 'replicate');
    YH(:,i_YH) = temp(:);

end
clear temp




% w = const.
w = ones(size(Ax_s));
%w(1:145,:,:) = 0;
write3D(fName_w_small, w(:,:,1:DS:end), 'float32');
w = w(:);


%% ----------------- BHC ---------------------

% Solve LS || YH g - (Ax_s)||^2_W for g (gamma)
[gam_hat] = solveWeightedLS(YH, Ax_s(:), w);
disp(['gam_hat = [', num2str(gam_hat'), ']; % unnormalized'])


% Solve LS || (YH g)a - y_0 ||^2_W for a (alpha)
[alpha] = solveWeightedLS(YH*gam_hat, y_0(:), w);
gam = gam_hat*alpha;
disp(['gam = [', num2str(gam'), ']; % normalized'])



gam_Y = gam(indicesY);
gam_H = gam(indicesH);

disp(['gam_Y = [', num2str(gam_Y'), ']; % normalized'])
disp(['gam_H = [', num2str(gam_H'), ']; % normalized'])


bhc_sino = reshape(YH*gam,     size(y_0));

%% ----------------- Scatter Correction ---------------------


delta = bhc_sino - Ax_s*alpha;


%params
sigma_g = 4;
corrStren = 0.33;
delta_blur = imgaussfilt(delta, sigma_g, 'padding', 'replicate');

y_new = bhc_sino - corrStren * delta_blur;




write3D(fName_bhc_sino_small, bhc_sino(:,:,1:DS:end), 'float32');
write3D(fName_delta_small, delta(:,:,1:DS:end), 'float32');
write3D(fName_delta_blur_small, delta_blur(:,:,1:DS:end), 'float32');
write3D(fName_y_new_small, y_new(:,:,1:DS:end), 'float32');

save(fName_statFile,'gam_Y','gam_H', 'sigma_g', 'corrStren');


%% ----------------- permanent changes happen hereafter ---------------------
%% Compute new e and y
y_old = read3D(binaryFNames.sino, 'float32');
e = read3D(binaryFNames.errsino, 'float32');


Ax = y_old - e;
e = y_new - Ax;



%% output
write3D(binaryFNames.sino, y_new, 'float32');
write3D(binaryFNames.errsino, e, 'float32');

%end