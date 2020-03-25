
clear all
inversionMasterList_fName = '../../../../control/Recon_4D/invMasterList.txt';
inversionMasterList_fName = '../../../../control/Inversion/QGGMRF/invMasterList_single.txt';

plainParamsFile = '../../../plainParams/plainParams.sh';
mfilename = 'BHC_SC_multi';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../Modular_PreprocessingRoutines'));
addpath(fullfile(mfilepath,'../../../Modular_MatlabRoutines'));

%% ----------------- params ------------------------------------------------------------------
clear par
par.applyCorrection = 0;

par.DS = 20;
par.isEnergyDomain = 1;
par.Ax_s_scaler = 0.75; % tau
par.exp_range = [0.5 1 2];
par.power_H = [ 1   1   1];
par.sigma_H = [ 50  100 200];
par.segmentationMethod = 'threshold'; % 'threshold', 'adaptive'
par.segmentationThreshold = 0.075;

%% ----------------- File Names and such -----------------------------------------------------
disp(' --- Read File Names ---')

inversionMasterList = readFileList(inversionMasterList_fName);
binaryFNames = readBinaryFNames( inversionMasterList{1}, plainParamsFile );

baseFolder = [fileparts(binaryFNames.recon), '/BHC'];
baseFolder_small = [fileparts(binaryFNames.recon), '/BHC_small'];
clear binaryFNames

fName_stats = [baseFolder_small, '/_stats.mat'];

fName_x =           [baseFolder,        '/x.recon'];
fName_x_small =     [baseFolder_small,  '/x_small.recon'];

fName_x_s =           [baseFolder,        '/x_s.recon'];
fName_x_s_small =     [baseFolder_small,  '/x_s_small.recon'];

for i = 1:length(inversionMasterList)

    suffix = sprintf('_%02d', i-1);

    binaryFNames{i} = readBinaryFNames( inversionMasterList{i}, plainParamsFile );

    fName_Ax_s{i} =                 [baseFolder,        '/Ax_s',         suffix, '.sino'];
    fName_Ax_s_small{i} =           [baseFolder_small,  '/Ax_s_small',   suffix, '.sino'];

    fName_y_new{i} =                [baseFolder,        '/y_new',         suffix, '.sino'];
    fName_y_new_small{i} =          [baseFolder_small,  '/y_new_small',   suffix, '.sino'];

    fName_y_0{i} =                  [baseFolder,        '/y_0',         suffix, '.sino'];
    fName_y_0_small{i} =            [baseFolder_small,  '/y_0_small',   suffix, '.sino'];

    fName_y_new_tilde{i} =          [baseFolder,        '/y_new_tilde',         suffix, '.sino'];
    fName_y_new_tilde_small{i} =    [baseFolder_small,  '/y_new_tilde_small',   suffix, '.sino'];

    fName_Ax_s_tilde{i} =           [baseFolder,        '/Ax_s_tilde',         suffix, '.sino'];
    fName_Ax_s_tilde_small{i} =     [baseFolder_small,  '/Ax_s_tilde_small',   suffix, '.sino'];

    fName_y_0_tilde{i} =            [baseFolder,        '/y_0_tilde',         suffix, '.sino'];
    fName_y_0_tilde_small{i} =      [baseFolder_small,  '/y_0_tilde_small',   suffix, '.sino'];

    fName_Yga{i} =                  [baseFolder,        '/Yga',         suffix, '.sino'];
    fName_Yga_small{i} =            [baseFolder_small,  '/Yga_small',   suffix, '.sino'];

    fName_Hmu{i} =                  [baseFolder,        '/Hmu',         suffix, '.sino'];
    fName_Hmu_small{i} =            [baseFolder_small,  '/Hmu_small',   suffix, '.sino'];

end


%% ----------------- AVG ---------------------------------------------------------------------
disp(' --- Averaging ---')

x = averageInversionRecons(plainParamsFile, inversionMasterList_fName);

write3D(fName_x, x, 'float32');
write3D(fName_x_small, x(1:par.DS:end,:,:), 'float32');
clear x

%% ----------------- Segmentation ------------------------------------------------------------
disp(' --- Segmentation ---')

x = read3D(fName_x, 'float32');

x_s = CustomSegment3D(x, par);

write3D(fName_x_s, x_s, 'float32');
write3D(fName_x_s_small, x_s(1:par.DS:end,:,:), 'float32');

clear x_s x




%% ----------------- Forward projection --------------------------------------------------
disp(' --- Forward Projection ---')
for i = 1:length(inversionMasterList)



    %% Forward project x_s -> Ax_s
    command = ['bash ../../../Inversion/run/forwardProject.sh -M ', inversionMasterList{i}, ' -i ', fName_x_s, ' -o ', fName_Ax_s{i}];
    [exit_status, commandLineOutput] = system(command);
    if(exit_status==1)
        error('Error with system command');
    end

    Ax_s = read3D(fName_Ax_s{i}, 'float32');

    write3D(fName_Ax_s{i}, Ax_s, 'float32');
    write3D(fName_Ax_s_small{i}, Ax_s(:,:,1:par.DS:end), 'float32');

end
clear Ax_s


%% ----------------- Forward projection --------------------------------------------------
disp(' --- Scaling Forward Projection ---')
for i = 1:length(inversionMasterList)

    Ax_s = read3D(fName_Ax_s{i}, 'float32');
    y_0 = read3D(binaryFNames{i}.origSino, 'float32');
    
    Ax_s = Ax_s * solveWeightedLS(Ax_s(:), y_0(:), ones(size(y_0(:))));
    Ax_s = Ax_s * par.Ax_s_scaler;

    write3D(fName_y_0{i}, y_0, 'float32');
    write3D(fName_y_0_small{i}, y_0(:,:,1:par.DS:end), 'float32');

    write3D(fName_Ax_s{i}, Ax_s, 'float32');
    write3D(fName_Ax_s_small{i}, Ax_s(:,:,1:par.DS:end), 'float32');

end
clear Ax_s

%% ----------------- BHC+SC --------------------------------------------
disp(' --- BHC+SC ---')
for i = 1:length(inversionMasterList)

    y_0 = read3D(binaryFNames{i}.origSino, 'float32');
    Ax_s = read3D(fName_Ax_s{i}, 'float32');


    y_new = zeros(size(y_0));
    y_new_tilde = zeros(size(y_0));
    Ax_s_tilde = zeros(size(y_0));
    y_0_tilde = zeros(size(y_0));
    Yga = zeros(size(y_0));
    Hmu = zeros(size(y_0));

    for i_view = 1:size(y_0, 3);

        [y_new(:,:,i_view), y_new_tilde(:,:,i_view), Ax_s_tilde(:,:,i_view), y_0_tilde(:,:,i_view), Yga(:,:,i_view), Hmu(:,:,i_view)] = BHC_SC_atomic(Ax_s(:,:,i_view), y_0(:,:,i_view), par);

    end

    write3D(fName_y_new{i}, y_new, 'float32');
    write3D(fName_y_new_small{i}, y_new(:,:,1:par.DS:end), 'float32');

    write3D(fName_y_new_tilde{i}, y_new_tilde, 'float32');
    write3D(fName_y_new_tilde_small{i}, y_new_tilde(:,:,1:par.DS:end), 'float32');

    write3D(fName_Ax_s_tilde{i}, Ax_s_tilde, 'float32');
    write3D(fName_Ax_s_tilde_small{i}, Ax_s_tilde(:,:,1:par.DS:end), 'float32');

    write3D(fName_y_0_tilde{i}, y_0_tilde, 'float32');
    write3D(fName_y_0_tilde_small{i}, y_0_tilde(:,:,1:par.DS:end), 'float32');

    write3D(fName_Yga{i}, Yga, 'float32');
    write3D(fName_Yga_small{i}, Yga(:,:,1:par.DS:end), 'float32');

    write3D(fName_Hmu{i}, Hmu, 'float32');
    write3D(fName_Hmu_small{i}, Hmu(:,:,1:par.DS:end), 'float32');


    if(par.applyCorrection==1)
        %% ----------------- permanent changes happen hereafter ----------------------------------
        %% Compute new e and y
        disp(' --- Correcting Binary Files ---')
        y_old = read3D(binaryFNames{i}.sino, 'float32');
        e = read3D(binaryFNames{i}.errsino, 'float32');


        Ax = y_old - e;
        e = y_new - Ax;



        %% output
        write3D(binaryFNames{i}.sino, y_new, 'float32');
        write3D(binaryFNames{i}.errsino, e, 'float32');
    end
end

save(fName_stats,'par');



%end